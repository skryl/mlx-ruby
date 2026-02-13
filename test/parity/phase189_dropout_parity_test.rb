# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase189DropoutParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_dropout_probability_validation_and_eval_passthrough
    assert_raises(ArgumentError) { MLX::NN::Dropout.new(-0.01) }
    assert_raises(ArgumentError) { MLX::NN::Dropout.new(1.0) }

    layer = MLX::NN::Dropout.new(0.25)
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    layer.eval
    y = layer.call(x)

    assert_same x, y
  end

  def test_dropout_training_scales_kept_values
    MLX::Core.random_seed(123)
    layer = MLX::NN::Dropout.new(0.5)
    layer.train(true)

    x = MLX::Core.array([[2.0, 4.0], [6.0, 8.0]], MLX::Core.float32)
    y = layer.call(x).to_a
    expected_scaled = [[4.0, 8.0], [12.0, 16.0]]

    y.flatten.zip(expected_scaled.flatten).each do |got, scaled|
      assert_includes [0.0, scaled], got
    end
  end

  def test_dropout2d_channelwise_masking_and_ndim_validation
    layer = MLX::NN::Dropout2d.new(0.5)
    bad = MLX::Core.full([2, 2], 1.0, MLX::Core.float32)
    assert_raises(ArgumentError) { layer.call(bad) }

    MLX::Core.random_seed(123)
    layer.train(true)
    x = MLX::Core.full([1, 2, 3, 4], 1.0, MLX::Core.float32)
    y = layer.call(x).to_a

    4.times do |channel|
      values = []
      2.times do |h|
        3.times do |w|
          values << y[0][h][w][channel]
        end
      end
      values.each { |v| assert_in_delta values.first, v, 1e-6 }
      assert_includes [0.0, 2.0], values.first
    end
  end

  def test_dropout3d_channelwise_masking_and_ndim_validation
    layer = MLX::NN::Dropout3d.new(0.5)
    bad = MLX::Core.full([2, 2, 2], 1.0, MLX::Core.float32)
    assert_raises(ArgumentError) { layer.call(bad) }

    MLX::Core.random_seed(321)
    layer.train(true)
    x = MLX::Core.full([1, 2, 2, 2, 3], 1.0, MLX::Core.float32)
    y = layer.call(x).to_a

    3.times do |channel|
      values = []
      2.times do |d|
        2.times do |h|
          2.times do |w|
            values << y[0][d][h][w][channel]
          end
        end
      end
      values.each { |v| assert_in_delta values.first, v, 1e-6 }
      assert_includes [0.0, 2.0], values.first
    end
  end
end
