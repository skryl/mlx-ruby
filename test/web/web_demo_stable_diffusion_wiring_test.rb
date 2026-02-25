# frozen_string_literal: true

require_relative "../support/test_helper"

class WebDemoStableDiffusionWiringTest < Minitest::Test
  WEB_INDEX_HTML = File.join(RUBY_ROOT, "web", "index.html")
  WEB_TASK = File.join(RUBY_ROOT, "tasks", "web_task.rb")
  BUILD_DOCS_ACTION = File.join(RUBY_ROOT, ".github", "actions", "build-docs", "action.yml")
  STABLE_DIFFUSION_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "index.html")
  STABLE_DIFFUSION_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "main.js")

  def test_demo_index_links_to_stable_diffusion_demo
    source = File.read(WEB_INDEX_HTML)

    assert_includes source, 'href="./demo/stable_diffusion/"'
    assert_includes source, "Stable Diffusion Kanji"
  end

  def test_demo_index_disables_stable_diffusion_link_when_assets_are_missing
    source = File.read(WEB_INDEX_HTML)

    assert_includes source, 'id="card-stable-diffusion"'
    assert_includes source, 'id="stable-diffusion-note"'
    assert_includes source, "configureStableDiffusionCard"
    assert_includes source, 'new URL("./assets/stable_diffusion/meta.json", baseUrl)'
    assert_includes source, 'new URL("../assets/stable_diffusion/meta.json", baseUrl)'
    assert_includes source, "card.removeAttribute(\"href\")"
    assert_includes source, "Stable Diffusion assets are unavailable on this host."
  end

  def test_stable_diffusion_demo_points_at_stable_diffusion_assets
    main_source = File.read(STABLE_DIFFUSION_MAIN_JS)
    html_source = File.read(STABLE_DIFFUSION_INDEX_HTML)

    assert_includes html_source, "<title>Stable Diffusion Kanji: Web Demo</title>"
    assert_includes main_source, "const ASSET_ROOT_CANDIDATES = Array.from("
    assert_includes main_source, 'new URL("../../assets/stable_diffusion", import.meta.url)'
    assert_includes main_source, 'new URL("../assets/stable_diffusion", import.meta.url)'
    assert_includes main_source, 'new URL("./assets/stable_diffusion", import.meta.url)'
    assert_includes main_source, "TEXT_ENCODER_PATH = `${ASSET_ROOT}/text_encoder.onnx`"
    assert_includes main_source, "UNET_PATH = `${ASSET_ROOT}/unet.onnx`"
    assert_includes main_source, "VAE_DECODER_PATH = `${ASSET_ROOT}/vae_decoder.onnx`"
    assert_includes main_source, 'executionProviders: [provider]'
    assert_includes main_source, "options.externalData"
    assert_includes main_source, "external_data"
    assert_includes main_source, "Stable Diffusion demo assets are required to run this page."
  end

  def test_pages_build_action_exports_and_publishes_stable_diffusion_assets
    source = File.read(BUILD_DOCS_ACTION)
    assert_includes source, "WEB_ASSETS_TARGETS: gpt2,nanogpt,stable_diffusion"
    assert_includes source, "test -f web/assets/stable_diffusion/meta.json"
    assert_includes source, 'test -f "${site_dir}/demo/assets/stable_diffusion/meta.json"'
    refute_includes source, '--exclude "assets/stable_diffusion/"'
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

  def test_web_start_requires_gpt2_and_nanogpt_metadata_assets
    source = File.read(WEB_TASK)
    assert_includes source, 'File.join(WEB_ROOT, "assets", "gpt2", "meta.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "gpt2", "vocab.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "gpt2", "merges.txt")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "gpt2", "prompt.presets.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "nanogpt", "meta.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "nanogpt", "tokenizer.json")'
    assert_includes source, 'File.join(WEB_ROOT, "assets", "nanogpt", "prompt.presets.json")'
  end

  def test_stable_diffusion_demo_does_not_block_runtime_on_trained_flag
    source = File.read(STABLE_DIFFUSION_MAIN_JS)
    refute_includes source, "if (!modelMeta?.weights?.trained)"
    assert_includes source, "await Promise.all(["
    assert_includes source, "assetExists(TEXT_ENCODER_PATH)"
    assert_includes source, "assetExists(UNET_PATH)"
    assert_includes source, "assetExists(VAE_DECODER_PATH)"
  end
end
