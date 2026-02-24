# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class GraphIrNativeTimingTest < Minitest::Test
  EXPORT_SCRIPT = <<~RUBY.freeze
    require "json"
    ruby_root = ENV.fetch("MLX_TEST_RUBY_ROOT")
    $LOAD_PATH.unshift(File.join(ruby_root, "lib"))
    require "mlx"

    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    content = MLX::ONNX.export_onnx_json(fun, x, y: y, model_name: "timing_probe_case")
    payload = JSON.parse(content)
    puts payload.fetch("format")
  RUBY

  TRANSLATE_SCRIPT = <<~RUBY.freeze
    require "json"
    ruby_root = ENV.fetch("MLX_TEST_RUBY_ROOT")
    $LOAD_PATH.unshift(File.join(ruby_root, "lib"))
    require "mlx"

    payload = {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [1], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "y", "shape" => [1], "dtype" => "float32" }],
      "constants" => [],
      "nodes" => [{ "op" => "Add", "inputs" => ["x", "x"], "outputs" => ["y"], "arguments" => [] }]
    }
    content = MLX::ONNX.graph_ir_to_onnx_json(payload, model_name: "timing_probe_case")
    result = JSON.parse(content)
    puts result.fetch("format")
  RUBY

  def setup
    return if self.class.instance_variable_defined?(:@native_probe_extension_ready)

    previous_force = ENV["MLX_RUBY_FORCE_REBUILD"]
    if TestSupport.instance_variable_defined?(:@native_built)
      TestSupport.remove_instance_variable(:@native_built)
    end
    ENV["MLX_RUBY_FORCE_REBUILD"] = "1"
    TestSupport.build_native_extension!
    self.class.instance_variable_set(:@native_probe_extension_ready, true)
  ensure
    ENV["MLX_RUBY_FORCE_REBUILD"] = previous_force
  end

  def test_export_onnx_json_emits_native_probe_when_enabled
    stdout, stderr, status = Open3.capture3(
      {
        "MLX_IR_NATIVE_TIMING" => "1",
        "MLX_TEST_RUBY_ROOT" => RUBY_ROOT
      },
      RbConfig.ruby,
      "-e",
      EXPORT_SCRIPT,
      chdir: RUBY_ROOT
    )

    assert status.success?, "subprocess failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert_match(/\Aonnx_stub_v1\s*\z/, stdout)
    assert_match(/\[mlx\.onnx\.native\.timing\] export_onnx_json/, stderr)
    assert_match(/args_decode_ms=/, stderr)
    assert_match(/trace_export_ms=/, stderr)
    assert_match(/constants_capture_ms=/, stderr)
    assert_match(/lower_onnx_ms=/, stderr)
    assert_match(/json_dump_ms=/, stderr)
  end

  def test_export_onnx_json_does_not_emit_native_probe_when_disabled
    stdout, stderr, status = Open3.capture3(
      { "MLX_TEST_RUBY_ROOT" => RUBY_ROOT },
      RbConfig.ruby,
      "-e",
      EXPORT_SCRIPT,
      chdir: RUBY_ROOT
    )

    assert status.success?, "subprocess failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert_match(/\Aonnx_stub_v1\s*\z/, stdout)
    refute_match(/\[mlx\.onnx\.native\.timing\] export_onnx_json/, stderr)
  end

  def test_ir_to_onnx_json_emits_native_probe_when_enabled
    stdout, stderr, status = Open3.capture3(
      {
        "MLX_IR_NATIVE_TIMING" => "1",
        "MLX_TEST_RUBY_ROOT" => RUBY_ROOT
      },
      RbConfig.ruby,
      "-e",
      TRANSLATE_SCRIPT,
      chdir: RUBY_ROOT
    )

    assert status.success?, "subprocess failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert_match(/\Aonnx_stub_v1\s*\z/, stdout)
    assert_match(/\[mlx\.onnx\.native\.timing\] graph_ir_to_onnx_json/, stderr)
    assert_match(/parse_json_ms=/, stderr)
    assert_match(/lower_onnx_ms=/, stderr)
    assert_match(/json_dump_ms=/, stderr)
  end
end
