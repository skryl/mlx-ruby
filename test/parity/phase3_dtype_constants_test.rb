# frozen_string_literal: true

require_relative "test_helper"

class Phase3DtypeConstantsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_dtype_objects_and_categories
    assert defined?(MLX::Core::Dtype)
    assert_respond_to MLX::Core, :bool_
    assert_respond_to MLX::Core, :int32
    assert_respond_to MLX::Core, :float32
    assert_respond_to MLX::Core, :complex64

    assert_respond_to MLX::Core, :generic
    assert_respond_to MLX::Core, :number
    assert_respond_to MLX::Core, :integer
    assert_respond_to MLX::Core, :floating

    bool_dtype = MLX::Core.bool_
    int_dtype = MLX::Core.int32
    float_dtype = MLX::Core.float32

    assert_instance_of MLX::Core::Dtype, bool_dtype
    assert_instance_of MLX::Core::Dtype, int_dtype
    assert_instance_of MLX::Core::Dtype, float_dtype

    assert_operator bool_dtype.size, :>=, 1
    assert_equal 4, int_dtype.size
    assert_equal 4, float_dtype.size

    assert_equal int_dtype, MLX::Core.int32
    refute_equal int_dtype, float_dtype

    assert_respond_to MLX::Core, :issubdtype
    assert MLX::Core.issubdtype(MLX::Core.int32, MLX::Core.integer)
    assert MLX::Core.issubdtype(MLX::Core.float32, MLX::Core.floating)
    refute MLX::Core.issubdtype(MLX::Core.float32, MLX::Core.integer)
  end

  def test_math_and_indexing_constants
    assert_respond_to MLX::Core, :pi
    assert_respond_to MLX::Core, :e
    assert_respond_to MLX::Core, :euler_gamma
    assert_respond_to MLX::Core, :inf
    assert_respond_to MLX::Core, :nan
    assert_respond_to MLX::Core, :newaxis

    assert_in_delta Math::PI, MLX::Core.pi, 1e-12
    assert_in_delta Math::E, MLX::Core.e, 1e-12

    assert MLX::Core.inf.infinite?
    assert MLX::Core.nan.nan?
    assert_nil MLX::Core.newaxis
  end
end
