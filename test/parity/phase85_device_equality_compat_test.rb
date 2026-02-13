# frozen_string_literal: true

require_relative "test_helper"

class Phase85DeviceEqualityCompatTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_device_compares_equal_to_device_type_symbols
    cpu = MLX::Core::Device.new(MLX::Core::DeviceType.cpu)
    gpu = MLX::Core::Device.new(MLX::Core::DeviceType.gpu)

    assert_equal true, (cpu == :cpu)
    assert_equal true, (gpu == :gpu)
    assert_equal false, (cpu == :gpu)
    assert_equal false, (gpu == :cpu)
  end
end
