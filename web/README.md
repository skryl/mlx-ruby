# Web Assets

Browser-facing runtime assets live under this top-level `web/` directory.

Current contents:

- `onnx_webgpu_harness/`
  - `index.html`: browser UI for running an exported ONNX model.
  - `harness.js`: ONNX Runtime Web + WebGPU/wasm fallback harness logic.
  - `browser_smoke.mjs`: Playwright-based smoke runner used by
    `MLX::ONNX.smoke_test_onnx_webgpu_harness`.

The Ruby API `MLX::ONNX.export_onnx_webgpu_harness` copies these assets into an
output directory alongside `model.onnx` and harness metadata files.

For local smoke testing from this directory:

- Install dependencies: `npm install`
- Install Chromium for Playwright: `npx playwright install chromium`
- Run the smoke script directly: `npm run smoke -- --harness-dir /path/to/exported_harness --mock-ort`

Smoke runner flags:

- `--mock-ort`: replace ONNX Runtime with a deterministic mock module.
- `--local-ort` (default): serve `onnxruntime-web/dist/*` from local
  `web/node_modules` so smoke runs do not depend on external CDNs.
- `--no-local-ort`: disable local routing and allow the harness CDN import path.

Smoke telemetry payload (`onnx_webgpu_telemetry_v1`) includes provider/fallback
stats and `sample_outputs` from the measured run for parity assertions.
