# frozen_string_literal: true

require_relative "test_helper"

class Phase49DependsAndNpyIoTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_depends_for_single_array_and_array_list
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    dep = MLX::Core.array([10.0], MLX::Core.float32)

    out_single = MLX::Core.depends(x, dep)
    assert MLX::Core.array_equal(out_single, x)

    y = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)
    out_list = MLX::Core.depends([x, y], [dep])
    assert_equal 2, out_list.length
    assert MLX::Core.array_equal(out_list[0], x)
    assert MLX::Core.array_equal(out_list[1], y)
  end

  def test_save_and_load_npy
    arr = MLX::Core.array([[1.5, 2.5], [3.5, 4.5]], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "tensor.npy")
      MLX::Core.save(path, arr)

      loaded_inferred = MLX::Core.load(path)
      assert_equal arr.shape, loaded_inferred.shape
      assert MLX::Core.array_equal(arr, loaded_inferred)

      loaded_explicit = MLX::Core.load(path, "npy", false)
      assert MLX::Core.array_equal(arr, loaded_explicit)
    end
  end
end
