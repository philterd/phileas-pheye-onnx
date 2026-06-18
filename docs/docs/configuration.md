# Configuration

Local inference is enabled per PhEye filter by setting a `modelPath`. This is the single switch between remote (HTTP) and local (in-process) detection.

## Enabling local inference

Set `modelPath` on a PhEye filter to the [model directory](model-directory.md) you want to load. In a redaction policy:

```json
{
  "phEyeConfiguration": {
    "modelPath": "/models/ph-eye-pii-en-small",
    "labels": ["person"]
  }
}
```

- `modelPath`: the path to a local model directory. When set, Phileas runs inference in-process using this module. When omitted, Phileas calls a remote PhEye service as before.
- `labels`: the entity labels the model should detect, for example `person`.

In [PhiSQL](https://www.philterd.ai/phisql/), the equivalent is the `MODEL` clause:

```sql
DETECT PHEYE MODEL '/models/ph-eye-pii-en-small'
```

## How detection runs

With `modelPath` set, `PhEyeFilter` discovers this module's `LocalPhEyeDetectorProvider` through `java.util.ServiceLoader`, builds a `LocalPhEyeDetector` for the model directory, and runs inference locally. The detector returns spans; the filter then applies its per-label confidence thresholds and replacement strategies on top, exactly as for remote detections. See [How It Works](how-it-works.md) for the discovery mechanism and the inference pipeline.

## Thresholds

Confidence thresholds and replacement strategies are configured on the PhEye filter the same way regardless of where detection runs, so a policy written for remote PhEye keeps the same threshold and strategy settings when you switch it to a local model. Each published model has a recommended confidence threshold; use the value documented on the model you load.
