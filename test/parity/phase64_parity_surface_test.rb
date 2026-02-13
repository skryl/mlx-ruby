# frozen_string_literal: true

require "set"
require_relative "test_helper"

class Phase64ParitySurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_python_vs_ruby_binding_name_parity_is_empty
    python_defs = Dir.glob(File.join(REPO_ROOT, "python", "src", "*.cpp")).flat_map do |path|
      File.read(path).scan(/m\.def\(\s*(?:\n\s*)*"([^"]+)"/m).flatten
    end.to_set

    ruby_defs = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
                    .scan(/rb_define_singleton_method\(mCore,\s*"([^"]+)"/)
                    .flatten
                    .to_set

    missing = python_defs - ruby_defs
    assert_equal Set.new, missing
  end

  def test_stream_context_and_kernel_surface
    before = MLX::Core.default_device
    inside_type = nil

    MLX::Core.stream(MLX::Core.cpu) do
      inside_type = MLX::Core.default_device.type
    end

    after = MLX::Core.default_device
    assert_equal :cpu, inside_type
    assert_equal before, after

    begin
      kernel = MLX::Core.metal_kernel("k", ["x"], ["y"], "y[thread_position_in_grid.x]=x[thread_position_in_grid.x];")
      assert_instance_of(MLX::Core::Kernel, kernel)
    rescue RuntimeError => e
      assert_match(/metal|Metal|backend/i, e.message)
    end

    begin
      kernel = MLX::Core.cuda_kernel("k", ["x"], ["y"], "int i=blockIdx.x*blockDim.x+threadIdx.x;")
      assert_instance_of(MLX::Core::Kernel, kernel)
    rescue RuntimeError => e
      assert_match(/cuda|CUDA|backend/i, e.message)
    end
    assert_raises(RuntimeError) do
      x = MLX::Core.array([1.0], MLX::Core.float32)
      MLX::Core.precompiled_cuda_kernel(
        "k",
        "",
        [x],
        [[1]],
        [MLX::Core.float32],
        [],
        [1, 1, 1],
        [1, 1, 1]
      )
    end
  end
end
