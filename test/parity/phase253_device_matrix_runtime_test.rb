# frozen_string_literal: true

require_relative "test_helper"

class Phase253DeviceMatrixRuntimeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def test_device_env_cpu_selects_cpu_default
    stdout, stderr, status = run_default_device_probe("cpu")

    assert status.success?, "probe failed stderr=#{stderr}"
    assert_equal "cpu", stdout.strip
  end

  def test_device_env_gpu_selects_gpu_default_when_available
    skip("GPU backend unavailable") unless gpu_available?

    stdout, stderr, status = run_default_device_probe("gpu")

    assert status.success?, "probe failed stderr=#{stderr}"
    assert_equal "gpu", stdout.strip
  end

  def test_nested_array_construction_does_not_abort_on_gpu
    skip("GPU backend unavailable") unless gpu_available?

    script = <<~CODE
      $LOAD_PATH.unshift(#{File.join(RUBY_ROOT, "lib").inspect})
      require "mlx"

      MLX::Core.set_default_device(MLX::Core.gpu)
      x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]])

      puts x.dtype.name
      puts x.shape.inspect
    CODE

    stdout, stderr, status = Open3.capture3({ "DEVICE" => "gpu" }, "ruby", "-e", script, chdir: REPO_ROOT)

    assert status.success?, "gpu nested-array probe failed stderr=#{stderr}"
    lines = stdout.lines.map(&:strip)
    assert_equal "float32", lines[0]
    assert_equal "[2, 2]", lines[1]
  end

  private

  def gpu_available?
    return @gpu_available unless @gpu_available.nil?

    script = <<~CODE
      $LOAD_PATH.unshift(#{File.join(RUBY_ROOT, "lib").inspect})
      require "mlx"

      available = MLX::Core.respond_to?(:metal_is_available) && MLX::Core.metal_is_available
      puts available ? "true" : "false"
    CODE

    stdout, = Open3.capture2("ruby", "-e", script, chdir: REPO_ROOT)
    @gpu_available = stdout.strip == "true"
  end

  def run_default_device_probe(device)
    script = <<~CODE
      $LOAD_PATH.unshift(#{File.join(RUBY_ROOT, "lib").inspect})
      require "mlx"
      puts MLX::Core.default_device.type
    CODE

    Open3.capture3({ "DEVICE" => device }, "ruby", "-e", script, chdir: REPO_ROOT)
  end
end
