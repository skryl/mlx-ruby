# frozen_string_literal: true

require_relative "test_helper"

class Phase84StreamContextSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_stream_returns_context_object_without_block
    before = MLX::Core.default_device
    context = MLX::Core.stream(MLX::Core.cpu)

    assert context, "expected stream() to return context when no block is given"
    assert_respond_to context, :enter
    assert_respond_to context, :exit

    context.enter
    assert_equal :cpu, MLX::Core.default_device.type
    context.exit
    assert_equal before, MLX::Core.default_device
  end

  def test_stream_block_form_still_works
    before = MLX::Core.default_device
    inside = nil

    MLX::Core.stream(MLX::Core.cpu) do
      inside = MLX::Core.default_device.type
    end

    assert_equal :cpu, inside
    assert_equal before, MLX::Core.default_device
  end
end
