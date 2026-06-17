# phileas-pheye-onnx

Optional **local (on-device) GLiNER inference** for the Phileas PhEye filter, via ONNX Runtime.

Core [`phileas`](https://github.com/philterd/phileas) talks to a PhEye service over HTTP. This
module lets phileas run a GLiNER model in-process instead, with no PhEye server. It is a separate
artifact so that phileas core stays lightweight: only applications that want local inference pull
the native ONNX Runtime and tokenizer dependencies.

## How it plugs in

phileas core defines a small SPI in `ai.philterd.phileas.services.filters.ai.pheye`:

- `PhEyeDetector` — produces raw detections (`PhEyeSpan`) for a piece of text.
- `PhEyeDetectorProvider` — builds a local detector; discovered via `java.util.ServiceLoader`.

When a PhEye filter is configured with a `modelPath` (the policy field, or phisql's
`DETECT PHEYE ... MODEL '<path>'`), `PhEyeFilter` looks up a `PhEyeDetectorProvider` on the
classpath. This module registers `LocalPhEyeDetectorProvider` via
`META-INF/services`, so simply adding it as a dependency enables local inference. With no
`modelPath`, phileas uses the remote HTTP detector as before.

## Usage

Add the dependency alongside `phileas`:

```xml
<dependency>
    <groupId>ai.philterd</groupId>
    <artifactId>phileas-pheye-onnx</artifactId>
    <version>4.1.0-SNAPSHOT</version>
</dependency>
```

Point a PhEye filter at a local model directory (policy JSON):

```json
{
  "phEyeConfiguration": {
    "modelPath": "/models/ph-eye-pii-en-small",
    "labels": ["person"]
  }
}
```

The model directory is the layout produced by `ph-eye-model-training`'s `./hub/publish.sh`:

- `model.onnx` (or `onnx/model.onnx`) — the exported GLiNER model
- `tokenizer.json` — the HuggingFace fast tokenizer
- `gliner_config.json` — span width, max length, prompt tokens

## What it does (GLiNER 0.2.25 uni-encoder span port)

`LocalPhEyeDetector` reimplements the pipeline PhEye runs through the Python `gliner` library:

1. Split text into words (whitespace splitter, regex `\w+(?:[-_]\w+)*|\S`, Unicode-aware) with char offsets.
2. Build the prompt: `[<<ENT>> label]*  <<SEP>>  <text words>` (markerV0).
3. Tokenize prompt+text pre-split; build `words_mask` marking the first subtoken of each text word.
4. Enumerate candidate spans `[i, i+width]` for `width` in `0..max_width-1` (`max_width` is 12 for the PII model).
5. Run ONNX inputs `input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask`; read `logits [words, width, classes]`.
6. Decode: `sigmoid(logits) > threshold`, then greedy flat (non-overlapping, highest score first) selection.
7. Map selected word spans back to character offsets and return `PhEyeSpan`s.

The `PhEyeFilter` then applies its per-label thresholds and replacement strategies on top, exactly as for remote detections.

## Status and the parity gate

This is a faithful port of the reference algorithm, but it is **not yet validated end to end**:

- It has not been compiled against the ONNX Runtime and DJL tokenizer dependencies in this environment.
- It has **not** been parity-tested against the Python `gliner` model.

A redaction model that decodes spans incorrectly silently leaks names, so before production use the
following must be confirmed against an actual exported model (a parity test, `LocalPhEyeDetectorParityTest`,
should compare this detector to Python `gliner.predict_entities` on sample text and require exact agreement):

- The exported model's ONNX input dtypes/shapes (this code uses int64 tensors) and the `logits` output layout.
- The DJL `Encoding` accessors used (`getWordIds()`, pre-split `encode(String[])` behavior) match the model's tokenizer parity.
- Span ordering and the word-to-char offset mapping produce identical spans to the Python pipeline.

## License

Apache-2.0.
