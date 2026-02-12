# frozen_string_literal: true

require_relative "test_helper"

class Phase70CudaKernelSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_cuda_kernel_no_longer_not_implemented
    begin
      kernel = MLX::Core.cuda_kernel(
        "add_one",
        ["inp"],
        ["out"],
        "int elem = blockIdx.x * blockDim.x + threadIdx.x; if (elem < out_size) { out[elem] = inp[elem] + (T)1; }"
      )
      assert_instance_of MLX::Core::Kernel, kernel

      x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
      assert_raises(RuntimeError) do
        kernel.call(
          inputs: [x],
          output_shapes: [[4]],
          output_dtypes: [MLX::Core.float32],
          grid: [1, 1, 1],
          threadgroup: [64, 1, 1],
          template: [["T", MLX::Core.float32]]
        )
      end
    rescue RuntimeError => e
      assert_match(/cuda|CUDA|not available|backend/i, e.message)
    end
  end
end
