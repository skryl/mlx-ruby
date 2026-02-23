# frozen_string_literal: true

ENV["MLX_TEST_TIMEOUT"] = "120"

require "json"
require "open3"
require "rbconfig"
require "tempfile"
require_relative "test_helper"

class Phase321ExamplesOnnxCaptureHookParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    @repo_root = RUBY_ROOT
    @submodule_root = File.join(@repo_root, "submodules", "mlx-ruby-examples")
    @runner = File.join(@submodule_root, "benchmark", "runner.rb")
    @force_cpu = File.join(@submodule_root, "benchmark", "ruby", "force_cpu.rb")
    @capture_hook = File.join(@repo_root, "examples", "benchmark", "benchmark_mlx_examples", "onnx_capture_hook.rb")
  end

  def test_capture_hook_exports_non_empty_ir_for_examples_script
    skip "mlx-ruby-examples submodule is not available" unless File.exist?(@runner)

    Tempfile.create(["examples_onnx_capture", ".json"]) do |capture|
      env = {
        "MLX_BENCHMARK" => "1",
        "MLX_BENCHMARK_DEVICE" => "cpu",
        "MLX_BENCHMARK_DRYRUN" => "1",
        "MLX_EXAMPLES_SUBMODULE_ROOT" => @submodule_root,
        "MLX_EXAMPLES_ONNX_CAPTURE_FILE" => capture.path
      }
      command = [
        RbConfig.ruby,
        "-I#{File.join(@repo_root, 'lib')}",
        "-r", @force_cpu,
        "-r", @capture_hook,
        "mnist/test.rb"
      ]

      stdout, stderr, status = Open3.capture3(env, *command, chdir: @submodule_root)
      assert status.success?, "capture command failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"

      payload = JSON.parse(File.read(capture.path))
      graph = payload.fetch("payload")
      nodes = graph.fetch("nodes")

      assert_operator nodes.length, :>, 0, "expected non-empty graph capture, got #{nodes.length}"
      assert payload.fetch("expected_outputs").is_a?(Hash)
      assert payload.fetch("output_names").is_a?(Array)
    end
  end
end
