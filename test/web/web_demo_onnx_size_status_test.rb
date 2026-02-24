# frozen_string_literal: true

require_relative "../support/test_helper"

class WebDemoOnnxSizeStatusTest < Minitest::Test
  GPT2_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "gpt2", "index.html")
  GPT2_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "gpt2", "main.js")
  NANOGPT_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "nanogpt", "index.html")
  NANOGPT_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "nanogpt", "main.js")
  STABLE_DIFFUSION_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "index.html")
  STABLE_DIFFUSION_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "main.js")

  def test_all_demos_expose_onnx_size_status_pane
    [GPT2_INDEX_HTML, NANOGPT_INDEX_HTML, STABLE_DIFFUSION_INDEX_HTML].each do |path|
      source = File.read(path)
      assert_includes source, 'id="status-onnx-size"'
      assert_includes source, 'id="badge-onnx-size"'
      assert_includes source, "ONNX Size: loading..."
      assert_includes source, 'id="badge-parameters"'
      assert_includes source, "Parameters: loading..."
    end
  end

  def test_language_demos_compute_onnx_size_from_http_headers
    [GPT2_MAIN_JS, NANOGPT_MAIN_JS].each do |path|
      source = File.read(path)
      assert_includes source, "const onnxSizeBadge = document.getElementById(\"badge-onnx-size\")"
      assert_includes source, "const parametersBadge = document.getElementById(\"badge-parameters\")"
      assert_includes source, "const onnxSizeStatus = document.getElementById(\"status-onnx-size\")"
      assert_includes source, "setOnnxSizeText"
      assert_includes source, "setParameterText"
      assert_includes source, "headers.get(\"content-length\")"
      assert_includes source, "ONNX Size:"
      assert_includes source, "Parameters:"
      assert_includes source, "MODEL_PATH"
    end
  end

  def test_stable_diffusion_computes_stage_onnx_sizes
    source = File.read(STABLE_DIFFUSION_MAIN_JS)
    assert_includes source, "const onnxSizeBadge = document.getElementById(\"badge-onnx-size\")"
    assert_includes source, "const parametersBadge = document.getElementById(\"badge-parameters\")"
    assert_includes source, "const onnxSizeStatus = document.getElementById(\"status-onnx-size\")"
    assert_includes source, "setOnnxSizeText"
    assert_includes source, "setParameterText"
    assert_includes source, "headers.get(\"content-length\")"
    assert_includes source, "TEXT_ENCODER_PATH"
    assert_includes source, "UNET_PATH"
    assert_includes source, "VAE_DECODER_PATH"
    assert_includes source, "ONNX Size:"
    assert_includes source, "Parameters:"
  end
end
