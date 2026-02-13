# frozen_string_literal: true

require_relative "test_helper"

class Phase133PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_133_contract
    %i[glorot_normal glorot_uniform he_normal he_uniform sparse orthogonal].each { |n| assert_respond_to MLX::NN, n }
    assert_respond_to MLX::NN.orthogonal, :call
  end
end
