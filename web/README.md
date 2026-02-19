# Web Assets

Browser-facing runtime assets live under this top-level `web/` directory.

Current contents:

- `onnx_webgpu_harness/`
  - `index.html`: browser UI for running an exported ONNX model.
  - `harness.js`: ONNX Runtime Web + WebGPU/wasm fallback harness logic.
  - `browser_smoke.mjs`: Playwright-based smoke runner used by
    `MLX::Core.smoke_test_onnx_webgpu_harness`.
- `assets/`
  - Generated/exported model assets and metadata used by browser demos.
- `../tasks/web_assets_task/`
  - `export_cvae_decoder_assets.rb`: reproducible GraphIR -> ONNX exporter for
    the CVAE decoder demo model.
  - `export_gpt2_assets.rb`: loads Hugging Face GPT-2 weights through
    `examples/gpt2_example.rb` and exports GraphIR -> ONNX assets for the
    browser demo.
  - `export_nanogpt_assets.rb`: reproducible GraphIR -> ONNX exporter for the
    nanoGPT demo assets.
- `../tasks/training_task/`
  - `train_nanogpt_shakespeare.rb`: trains nanoGPT demo weights on
    `test/fixtures/karpathy.txt`.
- `demo/cvae/`
  - Browser demo app that loads `web/assets/cvae/model.onnx` and runs
    slider-driven latent decoding on WebGPU.
- `demo/gpt2/`
  - Dedicated GPT-2 demo page backed by
    `web/assets/gpt2/model.onnx` assets exported from `examples/gpt2_example.rb`.

The Ruby API `MLX::Core.export_onnx_webgpu_harness` copies these assets into an
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

## Demo quickstart

Start the demo server from repo root:

- `bundle exec rake web:start`

Then open:

- `http://127.0.0.1:3030/`
- CVAE demo link: `http://127.0.0.1:3030/demo/cvae/`
- GPT-2 demo link: `http://127.0.0.1:3030/demo/gpt2/`

Notes:

- `rake web:start` auto-generates both `web/assets/cvae/model.onnx`
  + `web/assets/nanogpt/model.onnx` +
  `web/assets/gpt2/model.onnx` if assets are missing.
- You can regenerate assets explicitly with `bundle exec rake web:assets`.
- nanoGPT export also requires trained artifacts:
  - `web/assets/nanogpt/weights.npz`
  - `web/assets/nanogpt/tokenizer.json`
  - `web/assets/nanogpt/training_config.json`
  - run explicitly with: `ruby tasks/web_assets_task/export_nanogpt_assets.rb`
- If nanoGPT trained artifacts are missing, export skips ONNX generation (no random-init fallback).
- GPT-2 demo export loads weights from Hugging Face and writes exported assets
  to `web/assets/gpt2/` (HF checkpoint files are cached under
  `web/assets/gpt2/weights/`).
- Default export repo is `openai-community/gpt2`.
- Override with:
  - `GPT2_HF_REPO=distilgpt2 bundle exec rake web:assets`

## Train demo models

Run model training with one task:

- `bundle exec rake web:train`
  - defaults to `nanogpt`
- `bundle exec rake "web:train[nanogpt]"`

Then regenerate assets:

- `bundle exec rake web:assets`

Useful overrides:

- nanoGPT:
  - `NANOGPT_STEPS=300 bundle exec rake web:train`

## Refresh GPT-2 demo assets

Download/refresh GPT-2 browser assets:

- `bundle exec rake web:assets`
