# frozen_string_literal: true

require_relative "test_helper"

class Phase69MetalKernelTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_metal_kernel_surface_and_call_contract
    x = MLX::Core.array([1.0, 2.0, 3.0, 4.0], MLX::Core.float32)
    begin
      kernel = MLX::Core.metal_kernel(
        "add_one",
        ["inp"],
        ["out"],
        "uint elem = thread_position_in_grid.x; out[elem] = inp[elem] + (T)1;"
      )
      assert_instance_of MLX::Core::Kernel, kernel

      if MLX::Core.metal_is_available
        outs = kernel.call(
          inputs: [x],
          output_shapes: [[4]],
          output_dtypes: [MLX::Core.float32],
          grid: [4, 1, 1],
          threadgroup: [4, 1, 1],
          template: [["T", MLX::Core.float32]]
        )
        assert_equal 1, outs.length
        assert_equal [2.0, 3.0, 4.0, 5.0], outs.first.to_a
      else
        assert_raises(RuntimeError) do
          kernel.call(
            inputs: [x],
            output_shapes: [[4]],
            output_dtypes: [MLX::Core.float32],
            grid: [4, 1, 1],
            threadgroup: [4, 1, 1],
            template: [["T", MLX::Core.float32]]
          )
        end
      end
    rescue RuntimeError => e
      assert_match(/metal|Metal|backend/i, e.message)
    end
  end
end
