# phileas-pheye-onnx Release Notes

Notable changes to phileas-pheye-onnx, most recent first.

Full changelogs for each release are available in the [GitHub releases](https://github.com/philterd/phileas-pheye-onnx/releases).

## Version 1.2.0 - Not yet released

## Version 1.1.0 - October 23, 2026

* Targets Phileas 4.5.0.
* Updates ONNX Runtime to 1.31.0 and DJL tokenizers to 0.39.0. A build that swaps in `onnxruntime_gpu` should use 1.31.0 to match.
* `PHEYE_ONNX_DEVICE` (or `pheye.onnx.device`) set to `cuda` or `cuda:<n>` runs the model on that GPU, with ONNX Runtime's GPU package (`onnxruntime_gpu`) in place of the CPU package and CUDA available; otherwise a warning is logged and the model runs on the CPU. The default is `cpu`.
* The model now loads on a host where DJL finds CUDA. DJL's tokenizer chose a CUDA build of its native library there and downloaded it from publish.djl.ai, which failed without network access and stopped the model loading. This module now sets `RUST_FLAVOR` to `cpu` unless it is set, so DJL loads the CPU build it bundles.
* The detector reports its model's `max_len` and `max_width` from `gliner_config.json` (`maxWords()` and `maxWidth()`), so Phileas 4.5.0's PhEye filter divides long text into windows the model reads whole.
* ONNX Runtime uses `Runtime.availableProcessors()` intra-op threads, which follows a container's CPU set and quota, with spinning off. Previously it started a thread per physical core of the host, which in a container limited to 4 of 6 cores more than doubled the time for a long text, and under a CPU quota tripled it or worse. Set `PHEYE_ONNX_THREADS` or `pheye.onnx.threads` to override the count.
* A model directory is loaded once per process and shared by every PhEye filter that names it. Previously each filter set loaded its own copy, so every policy, and every revision of one, held another copy of the model in memory.
* Text longer than 512 tokens is no longer truncated by the tokenizer, matching the Python gliner library. Previously the last words of such text got no tokens and could be reported as spurious detections. A word left without a token now fails the detection instead of being scored.

## Version 1.0.0 - June 21, 2026

* First release. Adds optional local, on-device GLiNER inference for the Phileas PhEye filter via ONNX Runtime, so Phileas can run a model in-process instead of calling a remote PhEye service. The detector is discovered through the Phileas `PhEyeDetectorProvider` SPI, so adding this module to the classpath enables local inference when a PhEye filter sets a `modelPath`. Targets Phileas 4.1.0.
