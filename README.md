# phileas-pheye-onnx

Optional **local (on-device) GLiNER inference** for the Phileas PhEye filter, via ONNX Runtime.

Core [`phileas`](https://github.com/philterd/phileas) talks to a PhEye service over HTTP. This
module lets phileas run a GLiNER model in-process instead, with no PhEye server. It is a separate
artifact so that phileas core stays lightweight: only applications that want local inference pull
the native ONNX Runtime and tokenizer dependencies.

**Documentation:** [philterd.github.io/phileas-pheye-onnx](https://philterd.github.io/phileas-pheye-onnx/).

## How it plugs in

phileas core defines a small SPI in `ai.philterd.phileas.services.filters.ai.pheye`:

- `PhEyeDetector`: produces raw detections (`PhEyeSpan`) for a piece of text.
- `PhEyeDetectorProvider`: builds a local detector, discovered via `java.util.ServiceLoader`.

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

`4.1.0-SNAPSHOT` is a development build published to the Maven Central snapshot repository, which is
not served from the default Maven Central repository. To resolve it, add the snapshot repository to
your build (this module's snapshot lives there; the `phileas` dependency itself resolves from the
default Maven Central repository):

```xml
<repositories>
    <repository>
        <id>central-portal-snapshots</id>
        <url>https://central.sonatype.com/repository/maven-snapshots/</url>
        <releases>
            <enabled>false</enabled>
        </releases>
        <snapshots>
            <enabled>true</enabled>
        </snapshots>
    </repository>
</repositories>
```

Snapshots are mutable and are periodically pruned, so pin a released version for anything you need to
reproduce. Once a `4.1.0` release is cut it will resolve from the default Maven Central repository,
with no extra repository configuration.

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

- `model.onnx` (or `onnx/model.onnx`): the exported GLiNER model
- `tokenizer.json`: the HuggingFace fast tokenizer
- `gliner_config.json`: span width, max length, prompt tokens

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

## Parity

A redaction model that decodes spans incorrectly silently leaks names, so this detector is
parity-tested before being trusted. Two tests cover it (see `src/test/`):

- `LocalPhEyeDetectorParityTest` runs the detector against a tiny synthetic ONNX fixture
  (`src/test/resources/gliner-fixture/`, real ONNX Runtime and the real DJL tokenizer) and asserts
  the deterministic spans. It verifies the tensor feed/fetch wiring, the words-mask and span
  enumeration, the sigmoid/threshold/greedy-non-overlap decode, and the word-to-character offset
  mapping. It runs in the normal build and skips cleanly if the fixture is absent.
- `LocalPhEyeDetectorRealModelParityTest` runs the detector against a real exported GLiNER model
  (set `PHILEAS_GLINER_MODEL_DIR`) and asserts its spans match the Python `gliner.predict_entities`
  reference exactly: same offsets and labels, scores within tolerance. It is skipped unless the
  model directory is provided.

Both pass. Verifying parity surfaced and fixed a real bug: `span_mask` is a boolean tensor in the
GLiNER ONNX signature, and the detector had been feeding it as int64, which ONNX Runtime rejects.
The detector now reproduces the Python reference on real weights.

## License

Apache-2.0.
