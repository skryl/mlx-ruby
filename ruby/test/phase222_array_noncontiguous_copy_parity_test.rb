# frozen_string_literal: true

require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase222ArrayNoncontiguousCopyParityTest < Minitest::Test
  PY_NONCONTIG_SCRIPT = <<~PY.freeze
    import numpy as np
    import sys

    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = base[:, ::2]
    np.save(sys.argv[1], view)
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_loads_numpy_noncontiguous_view_values
    Dir.mktmpdir do |dir|
      path = File.join(dir, "noncontig.npy")

      begin
        run_python!("python3", "-c", PY_NONCONTIG_SCRIPT, path)
      rescue RuntimeError => e
        skip("numpy unavailable: #{e.message}") if e.message =~ /No module named 'numpy'/
        raise
      end

      loaded = MLX::Core.load(path)
      assert_equal [4, 3], loaded.shape
      assert_equal [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0], [12.0, 14.0, 16.0], [18.0, 20.0, 22.0]], loaded.to_a
    end
  end

  def test_copy_like_state_is_not_shared_between_instances
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = x.__copy__
    z = y.__setitem__(1, 9.0)

    assert_equal [1.0, 2.0, 3.0], x.to_a
    assert_equal [1.0, 2.0, 3.0], y.to_a
    assert_equal [1.0, 9.0, 3.0], z.to_a
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
