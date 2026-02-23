# frozen_string_literal: true

require_relative "../test_helper"

class WebDemoWeightsGuardrailTest < Minitest::Test
  LANGUAGE_DEMO_MAIN_JS = {
    "gpt2" => File.join(RUBY_ROOT, "web", "demo", "gpt2", "main.js"),
    "nanogpt" => File.join(RUBY_ROOT, "web", "demo", "nanogpt", "main.js")
  }.freeze

  GPT2_EXPORTER = File.join(RUBY_ROOT, "tasks", "web_assets_task", "export_gpt2_assets.rb")
  STABLE_DIFFUSION_EXPORTER = File.join(
    RUBY_ROOT,
    "tasks",
    "web_assets_task",
    "export_stable_diffusion_assets.rb"
  )
  NANOGPT_EXPORTER = File.join(RUBY_ROOT, "tasks", "web_assets_task", "export_nanogpt_assets.rb")
  STABLE_DIFFUSION_EXAMPLE = File.join(RUBY_ROOT, "examples", "web", "stable_diffusion_example.rb")

  def test_language_demos_disable_generate_until_weights_are_ready
    LANGUAGE_DEMO_MAIN_JS.each_value do |path|
      source = File.read(path)
      assert(
        source.include?("setGenerationEnabled(false)") || source.include?("generateButton.disabled = true"),
        "#{path} should disable generate while weights/session are unavailable"
      )
    end
  end

  def test_language_demos_surface_missing_weights_message
    LANGUAGE_DEMO_MAIN_JS.each_value do |path|
      source = File.read(path)
      assert_includes source, "Weights: missing"
      refute_includes source, "Weights: random init"
    end
  end

  def test_nanogpt_exporter_no_longer_emits_random_init_mode
    source = File.read(NANOGPT_EXPORTER)
    refute_includes source, "random init"
    assert_includes source, "DEFAULT_HF_REPO_ID"
    assert_includes source, "NanoGptExample::NanoGptModel.download_hf_weights!"
    refute_includes source, "missing trained artifacts"
    refute_includes source, "weights.npz"
  end

  def test_gpt2_exporter_defaults_to_openai_community_repo
    source = File.read(GPT2_EXPORTER)
    assert_includes source, 'DEFAULT_HF_REPO_ID = "openai-community/gpt2"'
    refute_includes source, "hf-internal-testing/tiny-random-gpt2"
  end

  def test_gpt2_exporter_uses_direct_native_onnx_binary_export_flow
    source = File.read(GPT2_EXPORTER)
    assert_includes source, "MLX::ONNX.export_onnx"
    refute_includes source, "MLX::ONNX.export_graph_ir_json"
    refute_includes source, "ir_path"
    refute_includes source, "Open3.capture3"
    refute_includes source, "python"
  end

  def test_nanogpt_exporter_uses_direct_native_onnx_binary_export_flow
    source = File.read(NANOGPT_EXPORTER)
    assert_includes source, "MLX::ONNX.export_onnx"
    refute_includes source, "MLX::ONNX.export_graph_ir_json"
    refute_includes source, "ir.json"
    refute_includes source, "Open3.capture3"
    refute_includes source, "python"
  end

  def test_stable_diffusion_exporter_uses_native_binary_export_without_python
    exporter_source = File.read(STABLE_DIFFUSION_EXPORTER)
    assert_includes exporter_source, 'DEFAULT_HF_REPO_ID = StableDiffusionExample::HF_REPO_ID'
    assert_includes exporter_source, "StableDiffusionExample.download_hf_weights!"
    assert_includes exporter_source, "StableDiffusionExample.load_text_encoder_from_hf_directory"
    assert_includes exporter_source, "StableDiffusionExample.load_unet_from_hf_directory"
    assert_includes exporter_source, "StableDiffusionExample.load_autoencoder_from_hf_directory"
    assert_includes exporter_source, "MLX::ONNX.export_onnx"
    assert_includes exporter_source, "text_encoder.onnx"
    assert_includes exporter_source, "unet.onnx"
    assert_includes exporter_source, "vae_decoder.onnx"
    refute_includes exporter_source, "TinyUnetModel"
    refute_includes exporter_source, ".mlx_subset.npz"
    refute_includes exporter_source, "random init fallback"
    refute_includes exporter_source, "StableDiffusionPipeline.from_pretrained"
    refute_includes exporter_source, "torch.onnx.export"
    refute_includes exporter_source, "Open3.capture3"
    refute_includes exporter_source, "python"
    refute_includes exporter_source, "ir.json"
    refute_includes exporter_source, ".ir.json"
  end

  def test_stable_diffusion_defaults_to_larger_pretrained_checkpoint
    source = File.read(STABLE_DIFFUSION_EXAMPLE)
    assert_includes source, 'HF_REPO_ID = "Ksgk-fy/stable-diffusion-v1-5-smaller-unet-kanji_99"'
    refute_includes source, 'HF_REPO_ID = "Narsil/tiny-stable-diffusion-torch"'
  end
end
