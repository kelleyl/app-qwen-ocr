# Qwen OCR

Two-stage OCR app for video frames using a Qwen3.5 multimodal model. Stage 1 runs vision OCR on frames identified by an upstream `TimeFrame` app. Stage 2 (optional) runs a text-only post-processing prompt on each OCR result. Each stage emits its own MMIF view with `TextDocument`s aligned to their source.

## What this app produces

- **OCR view** (always): one `TextDocument` per processed frame, aligned by an `Alignment` annotation to its source `TimePoint` (or `TimeFrame` if no representative TimePoint is available).
- **Post-processing view** (only when `postPrompt` is set): one `TextDocument` per OCR result, aligned by an `Alignment` annotation to the corresponding OCR `TextDocument`.

To trace a post-processed text back to a frame:
1. Read an `Alignment` in the post-processing view → its `source` is an OCR `TextDocument` long-id (e.g. `v_2:td_5`).
2. Read the `Alignment` in the OCR view whose `target` is that long-id → its `source` is the `TimePoint`/`TimeFrame` long-id (e.g. `v_0:tp_1453`).

## System requirements

- Docker (or local Python with CUDA-capable GPU)
- GPU strongly recommended. Default model (`Qwen/Qwen3.5-2B`) needs ~5 GB peak VRAM with `transformers` BF16 inference. `Qwen3.5-4B` peaks at the same ~5 GB (Mixture-of-Experts: only active experts live on device).
- ~10 GB container image (torch + transformers + CUDA stack).

## User instructions

For general CLAMS app usage, see [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### Configurable runtime parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modelName` | string | `Qwen/Qwen3.5-2B` | HF model id or local path. Tested: `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`. |
| `ocrPrompt` | string | extract-all-text default | Stage 1 user prompt. |
| `ocrSystemPrompt` | string | (empty) | Stage 1 system prompt. |
| `postPrompt` | string | (empty) | Stage 2 user prompt. **Empty disables stage 2.** Supports a `{ocr_text}` placeholder which is replaced with the OCR output; if the placeholder is absent, the OCR text is appended after a blank line. |
| `postSystemPrompt` | string | (empty) | Stage 2 system prompt. |
| `frameType` | string (multivalued) | (empty = all labels) | TimeFrame label(s) to process. Repeat to include multiple, e.g. `--frameType Chyron --frameType Slate`. |
| `appUri` | string | `http://apps.clams.ai/swt-detection/` | Substring matched against the upstream view's `app` URI to locate the source TimeFrame view. |
| `batchSize` | integer | 8 | Inference batch size (both stages). |
| `allRepresentatives` | boolean | `false` | If true, process every representative TimePoint of each TimeFrame. Otherwise only the first. |
| `maxNewTokens` | integer | 200 | Max generated tokens per inference call. |
| `temperature` | number | 0.0 | Sampling temperature. `0.0` = greedy/deterministic. A small non-zero value (e.g. `0.1`) sometimes avoids greedy character-drop artifacts on proper nouns, but can introduce randomness. |
| `numBeams` | integer | 1 | Number of beams for beam-search decoding. `1` = greedy. Higher values explore more decoding paths; `4` with internal `early_stopping=True` can fix some character-drop artifacts at ~3× the runtime, but may also disturb attribute extraction. Mutually exclusive with `temperature > 0`. |
| `config` | string | (empty) | Optional path to a YAML config file (relative to the app directory) that provides any of the prompt parameters. CLI parameters take precedence over config values. |

### Running the app

#### Docker

```bash
docker build -t app-qwen-ocr -f Containerfile .
docker run --gpus all -p 5000:5000 \
  -v /path/to/hf-cache:/cache/huggingface \
  -v /path/to/videos:/data/videos:ro \
  app-qwen-ocr
```

The HF cache mount lets the model download once and persist across runs.

#### CLI (one-shot, local Python)

```bash
python3 cli.py \
  --frameType Chyron \
  --config config/chyron-default.yaml \
  --batchSize 4 \
  input.mmif output.mmif
```

#### REST (server mode)

```bash
python3 app.py --port 5000             # dev server
python3 app.py --port 5000 --production  # gunicorn
```

`POST /annotate` with the input MMIF as the request body.

### Configs

Bundled configs in `config/`:

- **`chyron-default.yaml`** — OCR prompt for chyron frames + cataloger-style post-processing prompt that converts raw chyron text into a structured "name as written / surname-first / attributes" block (see [the cataloging guidelines context](https://github.com/clamsproject/aapb-ann-cataid) for the format).

A config file is just YAML mapping any of these keys to multiline strings:
```yaml
ocr_prompt: |
  ...
ocr_system_prompt: |
  ...
post_prompt: |
  ...    # use {ocr_text} as a placeholder
post_system_prompt: |
  ...
```

CLI flags override config values; config values are only used when the CLI flag is empty.

## Output schema

For input MMIF `M`, each invocation appends:

**OCR view** (always):
```json
{
  "metadata": {"app": "http://apps.clams.ai/qwen-ocr/...", "stage": "ocr", "appConfiguration": {...}},
  "annotations": [
    {"@type": ".../TextDocument/v1", "properties": {"id": "v_X:td_1", "text": {"@value": "..."}}},
    {"@type": ".../Alignment/v1",    "properties": {"id": "v_X:al_1", "source": "<TimePoint long-id>", "target": "v_X:td_1"}},
    ...
  ]
}
```

**Post-processing view** (when `postPrompt` is set):
```json
{
  "metadata": {"app": "http://apps.clams.ai/qwen-ocr/...", "stage": "post-processing", "appConfiguration": {...}},
  "annotations": [
    {"@type": ".../TextDocument/v1", "properties": {"id": "v_Y:td_1", "text": {"@value": "..."}}},
    {"@type": ".../Alignment/v1",    "properties": {"id": "v_Y:al_1", "source": "v_X:td_1", "target": "v_Y:td_1"}},
    ...
  ]
}
```

You may also see `Annotation/v6` entries in the post-processing view — these are framework-generated lineage records that mirror the source's `document`/`origin`/`provenance` properties when an `Alignment` references a cross-view `TextDocument`. They don't affect alignment traversal.

## Resources

- [CLAMS Documentation](https://clams.ai)
- [MMIF Specification](https://mmif.clams.ai)
- [Qwen3.5 model card](https://huggingface.co/Qwen/Qwen3.5-2B)
