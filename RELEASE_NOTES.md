# phileas-pheye-onnx Release Notes

Notable changes to phileas-pheye-onnx, most recent first.

Full changelogs for each release are available in the [GitHub releases](https://github.com/philterd/phileas-pheye-onnx/releases).

## Version 1.0.0 - June 21, 2026

* First release. Adds optional local, on-device GLiNER inference for the Phileas PhEye filter via ONNX Runtime, so Phileas can run a model in-process instead of calling a remote PhEye service. The detector is discovered through the Phileas `PhEyeDetectorProvider` SPI, so adding this module to the classpath enables local inference when a PhEye filter sets a `modelPath`. Targets Phileas 4.1.0.
