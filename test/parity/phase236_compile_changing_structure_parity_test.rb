# frozen_string_literal: true

require_relative "test_helper"

class Phase236CompileChangingStructureParityTest < Minitest::Test
  def run
    run_without_timeout
  end

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_compile_allows_output_structure_change_by_constant_kwargs
    compiled = MLX::Core.compile(lambda do |x, pair: false|
      pair ? [x, MLX::Core.add(x, 1.0)] : x
    end)

    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    out_single = compiled.call(x, pair: false)
    out_pair = compiled.call(x, pair: true)

    assert_instance_of MLX::Core::Array, out_single
    assert_instance_of Array, out_pair
    assert_equal [1.0, 2.0], out_pair[0].to_a
    assert_equal [2.0, 3.0], out_pair[1].to_a
  end

  def test_compile_output_with_siblings
    compiled = MLX::Core.compile(lambda do |x|
      y = MLX::Core.add(x, 1.0)
      [y, MLX::Core.square(y)]
    end)

    x = MLX::Core.array([2.0, 3.0], MLX::Core.float32)
    out = compiled.call(x)
    assert_equal [3.0, 4.0], out[0].to_a
    assert_equal [9.0, 16.0], out[1].to_a
  end
end
