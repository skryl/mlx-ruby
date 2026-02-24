# frozen_string_literal: true

require_relative "web_demo_integration_test_helper"

class StableDiffusionDemoIntegrationTest < Minitest::Test
  include WebDemoIntegrationTestHelper

  def test_stable_diffusion_demo_loads_metadata_and_generates
    ensure_web_runtime_dependencies!
    ensure_demo_assets!(
      "web/assets/stable_diffusion/text_encoder.onnx",
      "web/assets/stable_diffusion/unet.onnx",
      "web/assets/stable_diffusion/vae_decoder.onnx",
      "web/assets/stable_diffusion/meta.json",
      "web/assets/stable_diffusion/prompt.presets.json",
      "web/assets/stable_diffusion/scheduler_config.json",
      "web/assets/stable_diffusion/vocab.json",
      "web/assets/stable_diffusion/merges.txt"
    )

    with_web_demo_server do |base_url|
      result = probe_demo_page!(base_url, demo: "stable_diffusion")
      assert_equal "stable_diffusion", result.fetch("demo")
      assert_includes result.fetch("pre").fetch("model_status"), "stable_diffusion_kanji_web_demo"
      assert_match(/\AONNX Size:/, result.fetch("pre").fetch("onnx_size_status"))
      assert_match(/\AParameters: [0-9][0-9,]*\z/, result.fetch("pre").fetch("parameters_badge"))
      assert_match(/\AInference:/, result.fetch("post").fetch("timing_badge"))
      assert_match(/\AOutput:/, result.fetch("post").fetch("output_status"))
      assert_match(/\AStats:/, result.fetch("post").fetch("stats_status"))
    end
  end
end
