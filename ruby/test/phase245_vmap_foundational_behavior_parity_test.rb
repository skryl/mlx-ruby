# frozen_string_literal: true

require_relative "test_helper"

class Phase245VmapFoundationalBehaviorParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_vmap_unary_binary_tree_index_reduce_argreduce_mean
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([[10.0, 20.0], [30.0, 40.0]], MLX::Core.float32)

    unary = MLX::Core.vmap(->(v) { MLX::Core.square(v) })
    assert_equal [[1.0, 4.0], [9.0, 16.0]], unary.call(x).to_a

    binary = MLX::Core.vmap(->(a, b) { MLX::Core.add(a, b) })
    assert_equal [[11.0, 22.0], [33.0, 44.0]], binary.call(x, y).to_a

    tree_like = MLX::Core.vmap(->(v) { [MLX::Core.add(v, 1.0), MLX::Core.square(v)] })
    tree_out = tree_like.call(x)
    assert_equal [[2.0, 3.0], [4.0, 5.0]], tree_out[0].to_a
    assert_equal [[1.0, 4.0], [9.0, 16.0]], tree_out[1].to_a

    indexing = MLX::Core.vmap(->(v) { v.__getitem__(1) })
    assert_equal [2.0, 3.0], indexing.call(x).to_a

    reduce = MLX::Core.vmap(->(v) { MLX::Core.sum(v) })
    assert_equal [3.0, 7.0], reduce.call(x).to_a

    argreduce = MLX::Core.vmap(->(v) { MLX::Core.argmax(v, 0) })
    assert_equal [1, 1], argreduce.call(x).to_a

    mean = MLX::Core.vmap(->(v) { MLX::Core.mean(v) })
    assert_equal [1.5, 3.5], mean.call(x).to_a
  end

  def test_vmap_mismatch_input_sizes_raises
    binary = MLX::Core.vmap(->(a, b) { MLX::Core.add(a, b) })
    a = MLX::Core.array([[1.0, 2.0]], MLX::Core.float32)
    b = MLX::Core.array([[10.0, 20.0], [30.0, 40.0]], MLX::Core.float32)

    err = assert_raises(RuntimeError) { binary.call(a, b) }
    assert_match(/Inconsistent axis sizes/i, err.message)
  end
end
