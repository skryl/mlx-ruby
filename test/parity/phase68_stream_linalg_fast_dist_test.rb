# frozen_string_literal: true

require_relative "test_helper"

class Phase68StreamLinalgFastDistTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_linalg_and_fast_accept_stream_or_device
    cpu = MLX::Core.cpu
    stream = MLX::Core.new_stream(cpu)

    a = MLX::Core.array([[4.0, 7.0], [2.0, 6.0]], MLX::Core.float32)
    b = MLX::Core.array([1.0, 0.0], MLX::Core.float32)

    assert_equal [2], MLX::Core.solve(a, b, cpu).shape
    assert_equal [2, 2], MLX::Core.inv(a, stream).shape
    assert_equal [2], MLX::Core.eigvals(a, cpu).shape

    x = MLX::Core.array([[1.0, 2.0, 3.0]], MLX::Core.float32)
    w = MLX::Core.array([1.0, 1.0, 1.0], MLX::Core.float32)
    out = MLX::Core.rms_norm(x, w, 1e-5, stream)
    assert_equal [1, 3], out.shape
  end

  def test_distributed_collectives_accept_stream
    group = MLX::Core.init(false, "any")
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    cpu = MLX::Core.cpu
    stream = MLX::Core.new_stream(cpu)

    assert_equal [3], MLX::Core.all_sum(x, group, cpu).shape
    assert_equal [3], MLX::Core.all_max(x, group, stream).shape
    assert_equal [3], MLX::Core.all_min(x, group, cpu).shape
    assert_operator MLX::Core.all_gather(x, group, stream).shape[0], :>=, 3
    assert_equal [3], MLX::Core.sum_scatter(x, group, cpu).shape if group.size == 1
  end
end
