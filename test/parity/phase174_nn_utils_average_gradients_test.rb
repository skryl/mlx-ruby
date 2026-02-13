# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase174NnUtilsAverageGradientsTest < Minitest::Test
  GroupStub = Struct.new(:n) do
    def size
      n
    end
  end

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_average_gradients_is_noop_for_single_process_group
    grads = {
      "w" => MLX::Core.array([1.0, 2.0], MLX::Core.float32),
      "nested" => { "b" => MLX::Core.array([3.0], MLX::Core.float32) }
    }

    out = MLX::NN.average_gradients(grads, GroupStub.new(1), all_reduce_size: 0)

    assert_nested_close [1.0, 2.0], out.fetch("w").to_a
    assert_nested_close [3.0], out.fetch("nested").fetch("b").to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
