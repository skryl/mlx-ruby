# frozen_string_literal: true

require_relative "test_helper"

class Phase132PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_132_contract
    %i[constant normal uniform identity].each { |n| assert_respond_to MLX::NN, n }
    assert_respond_to MLX::NN.constant, :call
  end
end
