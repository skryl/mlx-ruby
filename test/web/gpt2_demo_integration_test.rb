# frozen_string_literal: true

require_relative "web_demo_integration_test_helper"

class Gpt2DemoIntegrationTest < Minitest::Test
  include WebDemoIntegrationTestHelper

  def test_gpt2_demo_loads_metadata_and_generates
    ensure_web_runtime_dependencies!
    ensure_demo_assets!(
      "web/assets/gpt2/model.onnx",
      "web/assets/gpt2/meta.json",
      "web/assets/gpt2/vocab.json",
      "web/assets/gpt2/merges.txt"
    )

    with_web_demo_server do |base_url|
      result = probe_demo_page!(base_url, demo: "gpt2")
      assert_equal "gpt2", result.fetch("demo")
      assert_includes result.fetch("pre").fetch("model_status"), "gpt2_ruby_web_demo"
      assert_match(/\AONNX Size:/, result.fetch("pre").fetch("onnx_size_status"))
      assert_match(/\AParameters: [0-9][0-9,]*\z/, result.fetch("pre").fetch("parameters_badge"))
      assert_match(/\AInference:/, result.fetch("post").fetch("timing_badge"))
    end
  end
end
