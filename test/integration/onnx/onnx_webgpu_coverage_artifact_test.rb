# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase319GraphIrWebgpuCoverageArtifactTest < Minitest::Test
  def self.current_test_timeout_seconds
    120
  end

  MODELS = %w[transformer cnn mlp rnn karpathy_gpt2].freeze

  def setup
    TestSupport.build_native_extension!
    @tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_onnx_webgpu_coverage_report.rb")
    @out_file = TestSupport.parity_generated_report_path("ir_webgpu_coverage.json")
  end

  def test_ir_webgpu_coverage_artifact_schema_and_baseline
    first = run_tool_and_read_payload
    assert_equal "ir_webgpu_coverage_v1", first.fetch("format")
    assert_equal MODELS, first.fetch("models")
    assert_equal MODELS.length, first.fetch("model_count")

    unsupported_ops_by_model = first.fetch("unsupported_ops_by_model")
    unsupported_nodes_by_model = first.fetch("unsupported_nodes_by_model")
    ready_by_model = first.fetch("ready_for_stub_conversion_by_model")
    total_nodes_by_model = first.fetch("total_nodes_by_model")

    MODELS.each do |model|
      assert unsupported_ops_by_model.key?(model), "missing unsupported_ops_by_model entry for #{model}"
      assert unsupported_nodes_by_model.key?(model), "missing unsupported_nodes_by_model entry for #{model}"
      assert ready_by_model.key?(model), "missing ready_for_stub_conversion_by_model entry for #{model}"
      assert total_nodes_by_model.key?(model), "missing total_nodes_by_model entry for #{model}"

      assert unsupported_ops_by_model.fetch(model).is_a?(Array)
      assert unsupported_nodes_by_model.fetch(model).is_a?(Integer)
      assert [true, false].include?(ready_by_model.fetch(model))
      assert total_nodes_by_model.fetch(model).is_a?(Integer)
    end

    # Phase 0 baseline is locked to zero unsupported ops for current benchmark fixtures.
    assert_equal [], first.fetch("unsupported_ops_union")
  end

  def test_ir_webgpu_coverage_artifact_is_deterministic
    first = run_tool_and_read_payload
    second = run_tool_and_read_payload
    assert_equal first, second
  end

  private

  def run_tool_and_read_payload
    env = {
      "IR_COVERAGE_FORCE_LOCAL" => "1",
      "IR_COVERAGE_MODELS" => MODELS.join(",")
    }
    stdout, stderr, status = Open3.capture3(env, "ruby", @tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(@out_file), "missing artifact at #{@out_file}"
    JSON.parse(File.read(@out_file))
  end
end
