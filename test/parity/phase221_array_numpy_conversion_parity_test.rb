# frozen_string_literal: true

require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase221ArrayNumpyConversionParityTest < Minitest::Test
  PY_NUMPY_SCRIPT = <<~PY.freeze
    import numpy as np
    import sys

    np.save(sys.argv[1], np.array([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32))
    np.save(sys.argv[2], np.array([1, 2, 3], dtype=np.int16))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_numpy_generated_npy_roundtrip_and_dtype_preservation
    Dir.mktmpdir do |dir|
      float_path = File.join(dir, "float.npy")
      int_path = File.join(dir, "int.npy")

      begin
        run_python!("python3", "-c", PY_NUMPY_SCRIPT, float_path, int_path)
      rescue RuntimeError => e
        skip("numpy unavailable: #{e.message}") if e.message =~ /No module named 'numpy'/
        raise
      end

      float_arr = MLX::Core.load(float_path)
      int_arr = MLX::Core.load(int_path)

      assert_equal [[1.25, 2.5], [3.75, 4.0]], float_arr.to_a
      assert_equal [1, 2, 3], int_arr.to_a
      assert_equal MLX::Core.float32, float_arr.dtype
      assert_equal MLX::Core.int16, int_arr.dtype
    end
  end

  def test_numpy_origin_array_participates_in_ruby_arithmetic
    Dir.mktmpdir do |dir|
      float_path = File.join(dir, "float.npy")
      int_path = File.join(dir, "int.npy")

      begin
        run_python!("python3", "-c", PY_NUMPY_SCRIPT, float_path, int_path)
      rescue RuntimeError => e
        skip("numpy unavailable: #{e.message}") if e.message =~ /No module named 'numpy'/
        raise
      end

      x = MLX::Core.load(float_path)
      y = MLX::Core.add(x, 1.0)
      assert_equal [[2.25, 3.5], [4.75, 5.0]], y.to_a
    end
  end

  private

  def run_python!(*cmd)
    stdout, stderr, status = Open3.capture3(*cmd)
    return if status.success?

    raise <<~MSG
      python command failed: #{cmd.join(" ")}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end
end
