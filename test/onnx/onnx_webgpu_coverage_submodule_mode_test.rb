# frozen_string_literal: true

ENV["MLX_TEST_TIMEOUT"] = "120"

require "json"
require "open3"
require_relative "test_helper"

class Phase320GraphIrWebgpuCoverageSubmoduleModeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    @tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_onnx_webgpu_coverage_report.rb")
    @out_file = File.join(RUBY_ROOT, "test", "parity", "reports", "ir_webgpu_coverage.json")
    @submodule_runner = File.join(RUBY_ROOT, "submodules", "mlx-ruby-examples", "benchmark", "runner.rb")
  end

  def test_submodule_mode_runs_requested_model_filter
    skip "mlx-ruby-examples submodule is not available" unless File.exist?(@submodule_runner)

    env = { "IR_COVERAGE_MODELS" => "mnist" }
    stdout, stderr, status = Open3.capture3(env, "ruby", @tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(@out_file), "missing artifact at #{@out_file}"

    payload = JSON.parse(File.read(@out_file))
    assert_equal "submodule", payload.fetch("mode")
    assert_equal ["mnist"], payload.fetch("models")
    assert_equal 1, payload.fetch("model_count")
    assert_equal({}, payload.fetch("errors_by_model"))
  end
end
