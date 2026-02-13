# frozen_string_literal: true

require_relative "test_helper"

class Phase141PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_141_contract
    %i[MultiHeadAttention TransformerEncoderLayer TransformerEncoder TransformerDecoderLayer TransformerDecoder Transformer].each { |n| assert MLX::NN.const_defined?(n) }
  end
end
