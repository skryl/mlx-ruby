# frozen_string_literal: true

require_relative "test_helper"

class Phase259FromDlpackZeroCopyPerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_from_dlpack_array_input_avoids_host_materialization
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    x.define_singleton_method(:to_a) do
      raise "from_dlpack(array) should not call to_a"
    end

    out = MLX::Core.from_dlpack(x)
    assert_same x, out
  end

  def test_from_dlpack_capsule_input_avoids_host_materialization
    x = MLX::Core.array([4.0, 5.0], MLX::Core.float32)
    x.define_singleton_method(:to_a) do
      raise "from_dlpack(capsule) should not call to_a"
    end

    capsule = x.__dlpack__
    out = MLX::Core.from_dlpack(capsule)
    assert_same x, out
  end
end
