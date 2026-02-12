# frozen_string_literal: true

require_relative "test_helper"

class Phase71PrecompiledCudaKernelTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_precompiled_cuda_kernel_surface
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)

    assert_raises(RuntimeError) do
      MLX::Core.precompiled_cuda_kernel(
        "dummy",
        "",
        [x],
        [[4]],
        [MLX::Core.float32],
        [],
        [1, 1, 1],
        [64, 1, 1]
      )
    end
  rescue NotImplementedError => e
    flunk("precompiled_cuda_kernel should not raise NotImplementedError: #{e.message}")
  end
end
