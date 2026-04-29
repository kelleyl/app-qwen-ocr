# app-qwen-ocr — developer notes

## Architecture

Two-stage processing within a single `ClamsApp`. One `transformers` model load is shared between stages.

1. **OCR stage**: vision inference on each frame. One TextDocument per frame, aligned to its source TimePoint (or TimeFrame if the upstream view has no representative TimePoint).
2. **Post-processing stage** (only when `postPrompt` is non-empty): text-only inference using the same loaded model, with each OCR result as input. One TextDocument per OCR result, aligned to the OCR TextDocument it was derived from.

Both stages call `_generate_batch()` with `enable_thinking=False` in the chat template (Qwen3.5 supports both vision and text from the same model with the same chat template).

## Output layout

Each invocation appends one or two views to the input MMIF:

```
view A (stage='ocr')                 view B (stage='post-processing')   [B only if postPrompt set]
├── TextDocument × N                  ├── TextDocument × N
├── Alignment × N                     ├── Alignment × N
│   source = upstream TP/TF           │   source = view A TextDocument long-id
│   target = view A TextDocument      │   target = view B TextDocument
                                      └── Annotation × N (auto-minted lineage records — ignored by alignment traversal)
```

Walking from post-processed text back to the original frame:
1. `Alignment` in view B → `source` is an OCR TextDocument long-id (e.g. `v_2:td_5`).
2. `Alignment` in view A whose `target` is that long-id → `source` is the TimePoint long-id.

## Configuration

Prompts can come from CLI flags or a YAML config file. CLI takes precedence. Recognized config keys:

```yaml
ocr_prompt: |
  ...
ocr_system_prompt: |
  ...
post_prompt: |
  ...   # use {ocr_text} as a placeholder
post_system_prompt: |
  ...
```

If `post_prompt` doesn't contain `{ocr_text}`, the OCR text is appended after a blank line.

## Frame extraction

For each TimeFrame matching `frameType`:
- If the TimeFrame has `representatives`, use the first representative TimePoint (or all reps if `allRepresentatives=true`).
- Otherwise fall back to `vdh.get_representative_framenum(mmif, tf)`.

Frames are deduplicated before extraction so adjacent TimeFrames sharing a representative don't trigger duplicate decodes.

## Decoding strategy

Default: greedy (`temperature=0.0`, `numBeams=1`). Deterministic and fast.

Two non-default modes available:
- `temperature > 0`: switches to nucleus sampling. A small value (0.1) sometimes escapes greedy character-drop bugs but may introduce randomness. Higher (0.7+) starts hallucinating.
- `numBeams > 1`: switches to beam search. Internally also enables `early_stopping=True` so generation cuts off when all beams hit EOS. `numBeams=4` fixed `"Lindsey, Claud"` → `"Lindsey, Claude"` on the eval set, at +5pp line-1 exact and -4pp attr F1, ~3× wall time. Mutually exclusive with `temperature > 0`.

Both options are wired through to `model.generate()` from the `_generate_batch` helper.

## Device handling

`__init__` picks the device in priority order:

1. **CUDA** if `torch.cuda.is_available()` → `dtype=bfloat16`
2. **MPS** (Apple Silicon) if available → `dtype=float16` (bfloat16 has spotty MPS coverage)
3. **CPU** fallback → `dtype=float32`

`pixel_values` are cast to `self.dtype` before generation. This makes the same code path runnable on a Mac for local development without touching the production deployment path.

Local Mac smoke test (M-series, 17 GB RAM, transformers via MPS, Qwen3.5-2B): 8 chyrons end-to-end in ~40 s. About 10× slower than CUDA but functional.

## VRAM

Both `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B` peak at ~5 GB during inference because the 4B variant uses sparse Mixture-of-Experts and only a subset of experts is active per forward pass. So the choice between 2B and 4B is largely about accuracy (and the on-disk + load-time difference, ~5 GB vs ~9 GB on disk) rather than runtime memory.

`Qwen/Qwen3.5-9B` is much heavier (~25 GB peak) and overkill for this task — 2B was within ~1 pp of 9B on the original benchmark.

## Local testing

```bash
cd ~/clams_apps/app-qwen-ocr

# Stage 1 only (OCR, no post-processing)
python3 cli.py --frameType Chyron \
  /path/to/swt_output.mmif /tmp/qwen_ocr.mmif

# Both stages, using bundled config
python3 cli.py --frameType Chyron --config config/chyron-default.yaml \
  /path/to/swt_output.mmif /tmp/qwen_ocr_full.mmif

# Force a specific GPU
CUDA_VISIBLE_DEVICES=2 python3 cli.py ...

# Try beam search for the formatter
python3 cli.py --frameType Chyron --config config/chyron-default.yaml --numBeams 4 \
  /path/to/swt_output.mmif /tmp/qwen_ocr_beams.mmif
```

The MMIF must already have `TimeFrame` annotations from an upstream SWT app (e.g. `app-swt-detection`).

## Container build

```bash
docker build -t app-qwen-ocr -f Containerfile .

# First-run will download the model from HF; persist with a cache mount:
docker run --gpus all -p 5000:5000 \
  -v $HOME/.cache/huggingface:/cache/huggingface \
  app-qwen-ocr
```

`Containerfile` uses `clams-python-opencv4-hf:latest`. HF cache is mapped to `/cache/huggingface` via `XDG_CACHE_HOME`.

## Known issues / gotchas

- **MMIF lineage Annotations.** When the post-processing view's `Alignment.source` references a TextDocument in the OCR view, MMIF auto-mints `Annotation/v6` entries in the post view that mirror the OCR TextDocument's `document`/`origin`/`provenance` properties. These are framework-internal lineage records, are not produced by app code, and don't affect alignment traversal. They're listed in the post view's `contains` map alongside TextDocument/Alignment.
- **Provenance gap.** `view.metadata.appConfiguration` is captured from the original CLI parameters, not from the merged config-file values. So if you supply prompts via `--config`, those prompts won't appear in the view metadata. The view is still functionally correct; just the recorded provenance is the CLI-flag state, not the resolved state.
- **Honorific drop at line 1 (small model).** With Qwen3.5-2B as the formatter, `Sen.` / `Rep.` / `Fr.` honorifics sometimes drop from the line-1 "name as written" form despite explicit prompt rules. Mitigated in `config/chyron-default.yaml` with extra negative examples; `Qwen3.5-9B` doesn't have this issue.
- **Greedy character drops.** Independent of size — Qwen3.5 (2B/4B/9B) all reproduce `"Claude" → "Claud"` patterns on greedy decode in surname-first re-emission. Beam search (`numBeams=4`) fixes name fields but trades some attribute fidelity. Default stays greedy because it's cleaner overall.
