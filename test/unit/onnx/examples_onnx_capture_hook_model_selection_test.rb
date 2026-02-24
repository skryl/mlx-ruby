# frozen_string_literal: true

require "json"
require "open3"
require "rbconfig"
require "tempfile"
require_relative "test_helper"

class Phase322ExamplesOnnxCaptureHookModelSelectionParityTest < Minitest::Test
  def self.current_test_timeout_seconds
    180
  end

  def setup
    TestSupport.build_native_extension!
    @repo_root = RUBY_ROOT
    @submodule_root = File.join(@repo_root, "submodules", "mlx-ruby-examples")
    @runner = File.join(@submodule_root, "benchmark", "runner.rb")
    @run_with_dryrun = File.join(@submodule_root, "benchmark", "ruby", "run_with_dryrun.rb")
    @force_cpu = File.join(@submodule_root, "benchmark", "ruby", "force_cpu.rb")
    @capture_hook = File.join(@repo_root, "examples", "benchmark", "benchmark_mlx_examples", "onnx_capture_hook.rb")
  end

  def test_cvae_capture_prefers_model_graph_not_trivial_probe
    payload = run_capture_for_model_script("cvae/test.rb")
    graph = payload.fetch("payload")

    assert_operator graph.fetch("nodes").length, :>, 10
    assert_operator graph.fetch("outputs").length, :>, 1
  end

  def test_segment_anything_capture_prefers_model_graph_not_trivial_probe
    payload = run_capture_for_model_script("segment_anything/test.rb")
    graph = payload.fetch("payload")

    assert_operator graph.fetch("nodes").length, :>, 10
  end

  def test_encodec_capture_exports_non_empty_graph
    payload = run_capture_for_model_script("encodec/test.rb")
    graph = payload.fetch("payload")

    assert_operator graph.fetch("nodes").length, :>, 10
  end

  private

  def run_capture_for_model_script(script)
    skip "mlx-ruby-examples submodule is not available" unless File.exist?(@runner)

    Tempfile.create(["examples_onnx_capture", ".json"]) do |capture|
      env = {
        "MLX_BENCHMARK" => "1",
        "MLX_BENCHMARK_DEVICE" => "cpu",
        "MLX_EXAMPLES_SUBMODULE_ROOT" => @submodule_root,
        "MLX_EXAMPLES_ONNX_CAPTURE_FILE" => capture.path
      }
      command = [
        RbConfig.ruby,
        "-I#{File.join(@repo_root, 'lib')}",
        "-r", @force_cpu,
        "-r", @capture_hook,
        @run_with_dryrun,
        script
      ]

      stdout, stderr, status = Open3.capture3(env, *command, chdir: @submodule_root)
      assert status.success?, "capture command failed for #{script}\nstdout:\n#{stdout}\nstderr:\n#{stderr}"

      payload = JSON.parse(File.read(capture.path))
      refute payload.key?("error"), "capture payload had error for #{script}: #{payload['error']}"
      return payload
    end
  end
end
