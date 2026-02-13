# frozen_string_literal: true

require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase223ArrayPickleStateParityTest < Minitest::Test
  PY_PICKLE_SCRIPT = <<~PY.freeze
    import pickle
    import numpy as np
    import sys

    payload = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    with open(sys.argv[1], "wb") as f:
      pickle.dump(payload, f)
    with open(sys.argv[1], "rb") as f:
      unpickled = pickle.load(f)
    np.save(sys.argv[2], unpickled)
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_state_protocol_roundtrip_preserves_values_and_dtype
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    state = x.__getstate__
    rebuilt = x.__setstate__(state)

    assert_equal x.to_a, rebuilt.to_a
    assert_equal x.dtype, rebuilt.dtype
  end

  def test_python_pickle_numpy_payload_can_be_loaded_through_npy
    Dir.mktmpdir do |dir|
      pickle_path = File.join(dir, "tensor.pkl")
      npy_path = File.join(dir, "tensor.npy")

      begin
        run_python!("python3", "-c", PY_PICKLE_SCRIPT, pickle_path, npy_path)
      rescue RuntimeError => e
        skip("python pickle/numpy unavailable: #{e.message}") if e.message =~ /No module named 'numpy'/
        raise
      end

      loaded = MLX::Core.load(npy_path)
      assert_equal [[5.0, 6.0], [7.0, 8.0]], loaded.to_a
      assert_equal MLX::Core.float32, loaded.dtype
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
