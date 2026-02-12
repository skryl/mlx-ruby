# frozen_string_literal: true

require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase65IoNpzLoadTest < Minitest::Test
  PY_BUILD_NPZ = <<~PY.freeze
    import os, sys, zipfile
    out_path = sys.argv[1]
    in_dir = sys.argv[2]
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
      for name in sorted(os.listdir(in_dir)):
        zf.write(os.path.join(in_dir, name), arcname=name)
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_load_npz_roundtrip_from_zipped_npy_files
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.array([[4.0, 5.0], [6.0, 7.0]], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      npy_dir = File.join(dir, "npy")
      Dir.mkdir(npy_dir)
      MLX::Core.save(File.join(npy_dir, "x.npy"), x)
      MLX::Core.save(File.join(npy_dir, "y.npy"), y)

      npz_path = File.join(dir, "weights.npz")
      run_python!("python3", "-c", PY_BUILD_NPZ, npz_path, npy_dir)

      loaded = MLX::Core.load(npz_path)
      assert_equal ["x", "y"], loaded.keys.sort
      assert MLX::Core.array_equal(x, loaded["x"])
      assert MLX::Core.array_equal(y, loaded["y"])

      err = assert_raises(ArgumentError) { MLX::Core.load(npz_path, "npz", true) }
      assert_match(/metadata not supported/i, err.message)
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
