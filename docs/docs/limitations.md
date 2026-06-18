# Limitations and Accuracy

This module is a faithful port of the GLiNER reference algorithm, but local inference has not yet been validated end to end against the Python `gliner` model. Treat it as functional but unverified until the parity checks below pass.

## Parity status

The module compiles and runs against ONNX Runtime and the DJL tokenizer, but it has not been parity-tested against the Python `gliner` model. A redaction model that decodes spans incorrectly can silently miss names, so before relying on local inference for anything that matters, confirm it produces the same spans as the reference pipeline.

A parity test (`LocalPhEyeDetectorParityTest`) should compare this detector to Python `gliner.predict_entities` on sample text and require exact agreement. Until that passes, prefer the remote PhEye detector for production, or validate the local detector's output on your own representative documents first.

The specific things to confirm against an actual exported model:

- The exported model's ONNX input dtypes and shapes (this code uses int64 tensors) and the `logits` output layout.
- The DJL tokenizer behavior used (`getWordIds()`, pre-split `encode(String[])`) matches the model's tokenizer parity.
- Span ordering and the word-to-character offset mapping produce identical spans to the Python pipeline.

## Accuracy

Detection with these models is probabilistic. A name detector will miss some names and flag some non-names, and accuracy depends on how close your text is to the data the model was trained on. Local inference does not change a model's accuracy: it runs the same model the remote PhEye service would, so the same calibration and recommended thresholds apply. Validate any model on your own text and set thresholds accordingly. You remain responsible for the personal data you process.

## Related

- [Phileas documentation](https://philterd.github.io/phileas/)
- [phileas-pheye-onnx on GitHub](https://github.com/philterd/phileas-pheye-onnx)
