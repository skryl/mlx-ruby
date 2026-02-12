# frozen_string_literal: true

require_relative "test_helper"

class Phase83DeviceTypeSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_device_type_surface_and_interop
    assert defined?(MLX::Core::DeviceType), "MLX::Core::DeviceType should exist"
    assert_respond_to MLX::Core::DeviceType, :cpu
    assert_respond_to MLX::Core::DeviceType, :gpu

    cpu_type = MLX::Core::DeviceType.cpu
    gpu_type = MLX::Core::DeviceType.gpu
    assert_equal :cpu, cpu_type
    assert_equal :gpu, gpu_type

    cpu_device = MLX::Core::Device.new(cpu_type)
    assert_equal :cpu, cpu_device.type

    assert_equal MLX::Core.device_count(:cpu), MLX::Core.device_count(cpu_type)
    assert_equal MLX::Core.device_count(:gpu), MLX::Core.device_count(gpu_type)
  end
end
