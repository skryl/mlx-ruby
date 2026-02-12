# frozen_string_literal: true

require "tmpdir"
require_relative "test_helper"

class Phase66IoSavezTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_savez_roundtrip_with_args_and_kwargs
    a = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    b = MLX::Core.array([[3.0, 4.0], [5.0, 6.0]], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "weights")
      MLX::Core.savez(path, a, b: b)

      loaded = MLX::Core.load(path + ".npz")
      assert_equal ["arr_0", "b"], loaded.keys.sort
      assert MLX::Core.array_equal(a, loaded["arr_0"])
      assert MLX::Core.array_equal(b, loaded["b"])
    end
  end

  def test_savez_compressed_roundtrip
    x = MLX::Core.array([10.0, 20.0, 30.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "compressed.npz")
      MLX::Core.savez_compressed(path, x: x)

      loaded = MLX::Core.load(path)
      assert_equal ["x"], loaded.keys
      assert MLX::Core.array_equal(x, loaded["x"])
    end
  end

  def test_savez_rejects_arr_i_keyword_collision
    x = MLX::Core.array([1.0], MLX::Core.float32)

    err = assert_raises(ArgumentError) do
      MLX::Core.savez("dummy", x, "arr_0" => x)
    end
    assert_match(/arr_0/, err.message)
  end
end
