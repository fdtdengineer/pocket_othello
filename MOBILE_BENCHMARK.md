# Mobile DQN Benchmark

After deploying a trained model, open the benchmark page on each target device:

```text
https://fdtdengineer.github.io/pocket_othello/benchmark.html
```

The page measures the standard opening position with the same Web Worker, ONNX model, WASM execution provider, observation encoding, and legal-action selection used by the game.

## Reported metrics

- model/session initialization time
- ONNX Runtime inference minimum, mean, median, p95, and maximum
- full worker round-trip minimum, mean, median, p95, and maximum
- selected legal move
- model byte size from export metadata
- browser user agent, logical cores, reported device memory, screen size, and isolation status

The first runs are discarded as configurable warmup iterations. Results can be copied as JSON for comparison.

## Suggested device matrix

Measure at least:

- one recent iPhone using Safari
- one older supported iPhone using Safari
- one recent Android device using Chrome
- one lower-end Android device using Chrome
- desktop Safari and Chrome as reference points

Use at least 30 measured iterations. For stable results, close other demanding applications and repeat the benchmark three times.

## Initial acceptance criteria

The following are project targets rather than guaranteed device capabilities:

- median worker round trip below 50 ms on recent phones
- p95 worker round trip below 100 ms on recent phones
- no illegal actions or inference failures
- one-time initialization below 2 seconds on a warm network/cache
- ONNX model below 1 MB

Do not quantize solely because the WASM runtime itself is large. Consider INT8 only if the trained FP32 model materially misses latency, memory, or download targets, then re-run both the benchmark and playing-strength evaluation.
