import argparse
import logging
import warnings
from pathlib import Path

import torch
import tqdm
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


class QwenOcr(ClamsApp):

    def __init__(self, model_name: str = None):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            self.logger.info(f"Device: cuda ({torch.cuda.get_device_name()}) dtype=bfloat16")
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16  # bfloat16 has spotty MPS coverage
            self.logger.info("Device: mps dtype=float16")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.logger.info("Device: cpu dtype=float32")

        model_path = model_name or DEFAULT_MODEL
        self.logger.info(f"Loading {model_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*preprocessor.json.*deprecated.*")
            self.processor = AutoProcessor.from_pretrained(model_path)

        # Decoder-only models need left-padding for batched generation
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype,
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded")

    def _appmetadata(self) -> AppMetadata:
        from metadata import appmetadata
        return appmetadata()

    # ---------- prompt / config helpers ----------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(__file__).parent / config_path
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _resolve_prompts(self, parameters: dict) -> dict:
        """Merge CLI parameters with config file.

        Resolution order: explicit CLI flag > config file value > metadata default.

        We can't just check ``parameters.get('ocrPrompt')`` because ClamsApp
        injects metadata defaults into the refined parameters dict. To
        distinguish "user explicitly passed --ocrPrompt" from "default was
        injected", we read from the ``#RAW#`` key which preserves the
        original user inputs only.
        """
        raw = parameters.get('#RAW#', {}) or {}

        def cli_value(key):
            v = raw.get(key)
            if isinstance(v, list):
                return v[0] if v else None
            return v

        cfg_path = cli_value('config') or parameters.get('config') or ''
        cfg = self._load_config(cfg_path) if cfg_path else {}

        def pick(cli_key, cfg_key):
            v = cli_value(cli_key)
            if v:
                return v
            if cfg.get(cfg_key):
                return cfg[cfg_key]
            return parameters.get(cli_key) or ''

        out = {
            'ocr_system': pick('ocrSystemPrompt', 'ocr_system_prompt'),
            'ocr_user': pick('ocrPrompt', 'ocr_prompt'),
            'post_system': pick('postSystemPrompt', 'post_system_prompt'),
            'post_user': pick('postPrompt', 'post_prompt'),
        }
        if not out['ocr_user']:
            raise ValueError("ocrPrompt must be provided (CLI or config)")
        return out

    def _sign_view_with_resolved_prompts(self, view, parameters, prompts):
        """Sign the view, then overwrite appConfiguration prompt fields with the
        post-config-merge values so the recorded provenance reflects what the
        app actually ran with (rather than the empty CLI defaults when prompts
        come from a YAML config).
        """
        self.sign_view(view, parameters)
        view.metadata.add_app_configuration('ocrPrompt', prompts['ocr_user'])
        view.metadata.add_app_configuration('ocrSystemPrompt', prompts['ocr_system'])
        view.metadata.add_app_configuration('postPrompt', prompts['post_user'])
        view.metadata.add_app_configuration('postSystemPrompt', prompts['post_system'])

    # ---------- inference ----------

    def _generate_batch(self, conversations: list, max_new_tokens: int,
                        temperature: float = 0.0, num_beams: int = 1) -> list:
        """Run a batch through the model and return the generated strings."""
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = inputs.to(self.device)
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.dtype)

        gen_kwargs = {'max_new_tokens': max_new_tokens}
        if temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
        elif num_beams > 1:
            gen_kwargs['num_beams'] = num_beams
            gen_kwargs['do_sample'] = False
            gen_kwargs['early_stopping'] = True
        else:
            gen_kwargs['do_sample'] = False

        generated = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated[:, prompt_len:]
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _ocr_batch(self, images: list, system_prompt: str, user_prompt: str,
                   max_new_tokens: int, temperature: float = 0.0, num_beams: int = 1) -> list:
        conversations = []
        for img in images:
            messages = []
            if system_prompt:
                messages.append({"role": "system",
                                 "content": [{"type": "text", "text": system_prompt}]})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_prompt},
                ],
            })
            conversations.append(messages)
        return self._generate_batch(conversations, max_new_tokens, temperature, num_beams)

    def _post_batch(self, ocr_texts: list, system_prompt: str, user_prompt_template: str,
                    max_new_tokens: int, temperature: float = 0.0, num_beams: int = 1) -> list:
        conversations = []
        for ocr_text in ocr_texts:
            if '{ocr_text}' in user_prompt_template:
                user_prompt = user_prompt_template.format(ocr_text=ocr_text)
            else:
                user_prompt = f"{user_prompt_template}\n\n{ocr_text}"
            messages = []
            if system_prompt:
                messages.append({"role": "system",
                                 "content": [{"type": "text", "text": system_prompt}]})
            messages.append({"role": "user",
                             "content": [{"type": "text", "text": user_prompt}]})
            conversations.append(messages)
        return self._generate_batch(conversations, max_new_tokens, temperature, num_beams)

    # ---------- target collection ----------

    @staticmethod
    def _matching_timeframes(mmif: Mmif, app_uri: str, labels: list) -> list:
        """Return TimeFrames from a view whose app URI matches and whose label is in labels."""
        timeframes = []
        for view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame):
            if app_uri and app_uri not in view.metadata.app:
                continue
            for tf in view.get_annotations(AnnotationTypes.TimeFrame):
                if not labels or tf.get_property('label') in labels:
                    timeframes.append(tf)
            if timeframes:
                break  # use first matching view
        return timeframes

    def _collect_tasks(self, mmif: Mmif, timeframes: list, all_reps: bool):
        """Build per-frame tasks from TimeFrames.

        Returns: list of (source_id, origin_id, framenum)
          source_id - the annotation that should be the alignment source (representative TP if available)
          origin_id - the parent TimeFrame id
          framenum  - frame number to extract
        """
        view_cache = {}

        def get_tp(tp_id: str):
            vid = tp_id.split(':')[0]
            if vid not in view_cache:
                try:
                    view_cache[vid] = mmif.get_view_by_id(vid)
                except Exception:
                    return None
            view = view_cache[vid]
            if not view:
                return None
            try:
                return view.annotations.get(tp_id)
            except Exception:
                return None

        def tp_to_framenum(tp_id):
            ann = get_tp(tp_id)
            if not ann or ann.at_type != AnnotationTypes.TimePoint:
                return None
            ms = vdh.convert_timepoint(mmif, ann, 'milliseconds')
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            return vdh.millisecond_to_framenum(video_doc, ms)

        tasks = []
        for tf in timeframes:
            reps = tf.get_property('representatives')
            if reps:
                rep_ids = reps if all_reps else [reps[0]]
                for rep_id in rep_ids:
                    fnum = tp_to_framenum(rep_id)
                    if fnum is None:
                        fnum = vdh.get_representative_framenum(mmif, tf)
                    tasks.append((rep_id, tf.long_id, fnum))
            else:
                fnum = vdh.get_representative_framenum(mmif, tf)
                tasks.append((tf.long_id, tf.long_id, fnum))
        return tasks

    # ---------- main ----------

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"parameters: {parameters}")
        prompts = self._resolve_prompts(parameters)
        labels = parameters.get('frameType') or []
        if isinstance(labels, str):
            labels = [labels] if labels else []
        labels = [l for l in labels if l]
        app_uri = parameters.get('appUri', '')
        batch_size = int(parameters.get('batchSize', 8))
        all_reps = bool(parameters.get('allRepresentatives', False))
        max_new_tokens = int(parameters.get('maxNewTokens', 200))
        temperature = float(parameters.get('temperature', 0.0))
        num_beams = int(parameters.get('numBeams', 1))

        # 1. Find target timeframes
        timeframes = self._matching_timeframes(mmif, app_uri, labels)
        if not timeframes:
            self.logger.warning(f"No timeframes found (appUri={app_uri!r} labels={labels})")
            return mmif
        self.logger.info(f"Found {len(timeframes)} timeframes to process")
        for tf in timeframes:
            tf.add_property('timeUnit', 'milliseconds')

        # 2. Build task list and extract frames
        tasks = self._collect_tasks(mmif, timeframes, all_reps)
        if not tasks:
            self.logger.warning("No tasks generated from timeframes")
            return mmif

        video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        unique_frames = sorted({t[2] for t in tasks if t[2] is not None})
        self.logger.info(f"Extracting {len(unique_frames)} unique frames")
        unique_images = vdh.extract_frames_as_images(video_doc, unique_frames, as_PIL=True)
        frame_to_image = dict(zip(unique_frames, unique_images))

        sortable = [(src, origin, fnum) for src, origin, fnum in tasks
                    if fnum in frame_to_image]
        sortable.sort(key=lambda t: t[2])

        # 3. Build the OCR view
        ocr_view: View = mmif.new_view()
        self._sign_view_with_resolved_prompts(ocr_view, parameters, prompts)
        ocr_view.metadata.set_additional_property('stage', 'ocr')
        ocr_view.new_contain(DocumentTypes.TextDocument)
        ocr_view.new_contain(AnnotationTypes.Alignment)

        # OCR results, paired with their tasks for later post-processing
        ocr_records = []  # list of dict(source, origin, td_long_id, text)

        for i in tqdm.tqdm(range(0, len(sortable), batch_size), desc='ocr'):
            batch = sortable[i:i + batch_size]
            images = [frame_to_image[fnum] for _, _, fnum in batch]
            outputs = self._ocr_batch(
                images, prompts['ocr_system'], prompts['ocr_user'], max_new_tokens, temperature, num_beams,
            )
            for (source_id, origin_id, _), text in zip(batch, outputs):
                # Create TD with text+mime via the canonical path; write
                # document/origin/provenance directly into properties to
                # bypass MMIF's _props_pending machinery (which would otherwise
                # spawn bookkeeping Annotation/v6 records in a downstream view).
                td = ocr_view.new_textdocument(text=text, mime='application/json')
                td.properties['document'] = video_doc.id
                td.properties['origin'] = origin_id
                td.properties['provenance'] = 'derived'
                align = ocr_view.new_annotation(AnnotationTypes.Alignment)
                align.add_property('source', source_id)
                align.add_property('target', td.long_id)
                ocr_records.append({
                    'source': source_id,
                    'origin': origin_id,
                    'td_long_id': td.long_id,
                    'text': text,
                })

        # 4. Optional post-processing view
        if prompts['post_user']:
            post_view: View = mmif.new_view()
            self._sign_view_with_resolved_prompts(post_view, parameters, prompts)
            post_view.metadata.set_additional_property('stage', 'post-processing')
            post_view.new_contain(DocumentTypes.TextDocument)
            post_view.new_contain(AnnotationTypes.Alignment)

            for i in tqdm.tqdm(range(0, len(ocr_records), batch_size), desc='post'):
                batch = ocr_records[i:i + batch_size]
                texts_in = [r['text'] for r in batch]
                texts_out = self._post_batch(
                    texts_in, prompts['post_system'], prompts['post_user'], max_new_tokens, temperature, num_beams,
                )
                for rec, text in zip(batch, texts_out):
                    td = post_view.new_textdocument(text=text, mime='application/json')
                    # origin = the original TimeFrame from the upstream SWT view
                    # (so a downstream consumer can trace post-text -> frame
                    # without walking through the OCR view).
                    td.properties['origin'] = rec['origin']
                    td.properties['provenance'] = 'derived'
                    # The Alignment links this post-TD to the OCR-TD it was
                    # derived from; no `document` property on the post-TD itself
                    # to keep this view minimal.
                    align = post_view.new_annotation(AnnotationTypes.Alignment)
                    align.add_property('source', rec['td_long_id'])
                    align.add_property('target', td.long_id)

        return mmif


def get_app():
    return QwenOcr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--model-name", default=None,
                        help="HuggingFace model id or local path")
    args = parser.parse_args()

    app = QwenOcr(model_name=args.model_name)
    http_app = Restifier(app, port=args.port)
    if args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
