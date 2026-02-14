# frozen_string_literal: true

require_relative "test_helper"

class Phase258TransformerCheckpointWrapperCachePerfTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_transformer_encoder_checkpoint_wrappers_are_not_rebuilt_per_call
    calls = 0
    trace = TracePoint.new(:call) do |tp|
      next unless tp.method_id == :checkpoint
      next unless tp.defined_class.equal?(MLX::NN.singleton_class)

      calls += 1
    end

    trace.enable do
      encoder = MLX::NN::TransformerEncoder.new(
        2,
        8,
        2,
        checkpoint: true,
        dropout: 0.0
      )

      x = MLX::Core.array([[[0.1] * 8, [0.2] * 8]], MLX::Core.float32)

      encoder.call(x, nil)
      encoder.call(x, nil)
    end

    assert_equal 2, calls
  end
end
