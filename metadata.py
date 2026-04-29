"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="Qwen OCR",
        description=(
            "Two-stage OCR + optional post-processing using a Qwen3.5 multimodal model. "
            "Stage 1 runs the OCR prompt against frames identified by an upstream "
            "TimeFrame app and emits TextDocuments aligned to those frames. "
            "Stage 2 (optional) runs a text-only post-processing prompt against each "
            "OCR result and emits a second view of TextDocuments aligned to the "
            "OCR TextDocuments."
        ),
        app_license="Apache 2.0",
        identifier="qwen-ocr",
        url="https://github.com/clamsproject/app-qwen-ocr",
    )

    metadata.add_input(DocumentTypes.VideoDocument)
    metadata.add_input(AnnotationTypes.TimeFrame)
    metadata.add_output(AnnotationTypes.Alignment)
    metadata.add_output(DocumentTypes.TextDocument)

    metadata.add_parameter(
        name='modelName', type='string', default='Qwen/Qwen3.5-2B',
        description='HuggingFace model name/path for the Qwen3.5 model. '
                    'Examples: Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B.',
    )
    metadata.add_parameter(
        name='ocrPrompt', type='string',
        default='Extract all visible on-screen text from this video frame, exactly as written. '
                'Preserve line breaks between distinct text regions. Output only the text.',
        description='User prompt for stage 1 (OCR). Sent with each frame image.',
    )
    metadata.add_parameter(
        name='ocrSystemPrompt', type='string', default='',
        description='Optional system prompt for stage 1 (OCR).',
    )
    metadata.add_parameter(
        name='postPrompt', type='string', default='',
        description='User prompt for stage 2 (post-processing). When non-empty, the app runs a '
                    'second text-only inference with the OCR output as input. The template '
                    'string {ocr_text} (if present) is replaced with the OCR text; otherwise '
                    'the OCR text is appended after a blank line.',
    )
    metadata.add_parameter(
        name='postSystemPrompt', type='string', default='',
        description='Optional system prompt for stage 2 (post-processing).',
    )
    metadata.add_parameter(
        name='frameType', type='string', default='', multivalued=True,
        description='TimeFrame label(s) to process. Only timeframes matching one of the specified '
                    'labels are run through OCR. Can be specified multiple times '
                    '(e.g. --frameType chyron --frameType slate).',
    )
    metadata.add_parameter(
        name='appUri', type='string',
        default='http://apps.clams.ai/swt-detection/',
        description='URI substring of the upstream app whose TimeFrame annotations to process.',
    )
    metadata.add_parameter(
        name='batchSize', type='integer', default=16,
        description='Number of items per inference batch. With Qwen3.5-2B BF16, '
                    'batch=16 peaks at ~6 GB VRAM. Reduce if memory-constrained.',
    )
    metadata.add_parameter(
        name='allRepresentatives', type='boolean', default=False,
        description='If True, process every representative TimePoint of each TimeFrame; '
                    'otherwise only the first representative.',
    )
    metadata.add_parameter(
        name='maxNewTokens', type='integer', default=80,
        description='Max new tokens generated per inference call (both stages). '
                    'Sized for typical chyron length (longest observed: ~22 tokens; '
                    'cap is ~3-4x that). Raise for longer credit-roll or slate text.',
    )
    metadata.add_parameter(
        name='temperature', type='number', default=0.0,
        description='Sampling temperature for both stages. 0.0 = greedy/deterministic. '
                    'A small non-zero value (e.g. 0.1) can avoid greedy-decoding artifacts '
                    'such as character drops on proper nouns.',
    )
    metadata.add_parameter(
        name='numBeams', type='integer', default=1,
        description='Number of beams for beam-search decoding (both stages). 1 = greedy. '
                    'Higher values explore more decoding paths and can avoid character-drop '
                    'artifacts on proper nouns at the cost of ~20% more inference time. '
                    'Mutually exclusive with temperature > 0.',
    )
    metadata.add_parameter(
        name='repetitionPenalty', type='number', default=1.0,
        description='Repetition penalty applied during generation (both stages). '
                    'Values > 1.0 discourage the model from repeating recently-emitted tokens. '
                    '1.0 = off (default — combined with the maxNewTokens cap, runaway loops are '
                    'already bounded). Raising to ~1.1 can shorten residual loop text further but '
                    'risks light hallucination at higher values.',
    )
    metadata.add_parameter(
        name='config', type='string', default='',
        description='Optional path to a YAML config file (relative to the app directory) that '
                    'can supply ocrPrompt / ocrSystemPrompt / postPrompt / postSystemPrompt. '
                    'CLI parameters take precedence over config values.',
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
