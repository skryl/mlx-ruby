# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase323ExamplesSubmoduleFullExportParityTest < Minitest::Test
  def self.current_test_timeout_seconds
    480
  end

  def setup
    TestSupport.build_native_extension!
    @tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_onnx_webgpu_coverage_report.rb")
    @out_file = TestSupport.parity_generated_report_path("ir_webgpu_coverage.json")
    @submodule_runner = File.join(RUBY_ROOT, "submodules", "mlx-ruby-examples", "benchmark", "runner.rb")
  end

  def test_all_submodule_models_are_webgpu_export_compatible
    skip "mlx-ruby-examples submodule is not available" unless File.exist?(@submodule_runner)

    expected_models = load_model_ids
    stdout, stderr, status = Open3.capture3("ruby", @tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(@out_file), "missing artifact at #{@out_file}"

    payload = JSON.parse(File.read(@out_file))
    assert_equal "submodule", payload.fetch("mode")
    assert_equal expected_models, payload.fetch("models")
    assert_equal expected_models.length, payload.fetch("model_count")
    assert_equal({}, payload.fetch("errors_by_model"))
    assert_equal [], payload.fetch("unsupported_ops_union")

    unsupported_nodes = payload.fetch("unsupported_nodes_by_model")
    expected_models.each do |model|
      assert_equal 0, unsupported_nodes.fetch(model), "#{model} had unsupported nodes"
    end
  end

  private

  def load_model_ids
    ids = []
    File.foreach(@submodule_runner) do |line|
      match = line.match(/^\s*id:\s*"([^"]+)",\s*$/)
      ids << match[1] if match
    end
    ids
  end
end
