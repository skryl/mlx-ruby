# frozen_string_literal: true

require_relative "../test_helper"

class WebDemoStableDiffusionWiringTest < Minitest::Test
  WEB_INDEX_HTML = File.join(RUBY_ROOT, "web", "index.html")
  WEB_TASK = File.join(RUBY_ROOT, "tasks", "web_task.rb")
  STABLE_DIFFUSION_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "index.html")
  STABLE_DIFFUSION_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "main.js")

  def test_demo_index_links_to_stable_diffusion_demo
    source = File.read(WEB_INDEX_HTML)

    assert_includes source, 'href="./demo/stable_diffusion/"'
    assert_includes source, "Stable Diffusion Tiny"
  end

  def test_stable_diffusion_demo_points_at_stable_diffusion_assets
    main_source = File.read(STABLE_DIFFUSION_MAIN_JS)
    html_source = File.read(STABLE_DIFFUSION_INDEX_HTML)

    assert_includes html_source, "<title>Stable Diffusion Tiny: Web Demo</title>"
    assert_includes main_source, 'const ASSET_ROOT = "../../assets/stable_diffusion"'
    assert_includes main_source, "const TEXT_ENCODER_PATH = `${ASSET_ROOT}/text_encoder.onnx`"
    assert_includes main_source, "const UNET_PATH = `${ASSET_ROOT}/unet.onnx`"
    assert_includes main_source, "const VAE_DECODER_PATH = `${ASSET_ROOT}/vae_decoder.onnx`"
    assert_includes main_source, 'executionProviders: [provider]'
    assert_includes main_source, "options.externalData"
    assert_includes main_source, "external_data"
  end

  def test_stable_diffusion_demo_exposes_multistep_generate_controls
    main_source = File.read(STABLE_DIFFUSION_MAIN_JS)
    html_source = File.read(STABLE_DIFFUSION_INDEX_HTML)

    assert_includes html_source, 'id="steps"'
    assert_includes html_source, 'id="guidance"'
    assert_includes html_source, "Generate Image"
    assert_includes main_source, "runMultiStepDenoise"
    assert_includes main_source, "for (let stepIndex = 0; stepIndex < steps; stepIndex += 1)"
  end

  def test_web_start_requires_stable_diffusion_model_asset
    source = File.read(WEB_TASK)
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "model.onnx")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "text_encoder.onnx")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "unet.onnx")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "vae_decoder.onnx")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "vocab.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "stable_diffusion", "merges.txt")'
  end
end
