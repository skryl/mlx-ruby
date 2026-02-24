# frozen_string_literal: true

require_relative "web_demo_integration_test_helper"

class NanogptDemoIntegrationTest < Minitest::Test
  include WebDemoIntegrationTestHelper

  def test_nanogpt_demo_loads_metadata_and_generates
    ensure_web_runtime_dependencies!
    ensure_demo_assets!(
      "web/assets/nanogpt/model.onnx",
      "web/assets/nanogpt/meta.json",
      "web/assets/nanogpt/tokenizer.json",
      "web/assets/nanogpt/prompt.presets.json"
    )

    with_web_demo_server do |base_url|
      result = probe_demo_page!(base_url, demo: "nanogpt")
      assert_equal "nanogpt", result.fetch("demo")
      assert_includes result.fetch("pre").fetch("model_status"), "nanogpt_shakespeare_web_demo"
      assert_match(/\AONNX Size:/, result.fetch("pre").fetch("onnx_size_status"))
      assert_match(/\AParameters: [0-9][0-9,]*\z/, result.fetch("pre").fetch("parameters_badge"))
      assert_match(/\AInference:/, result.fetch("post").fetch("timing_badge"))
    end
  end
end
