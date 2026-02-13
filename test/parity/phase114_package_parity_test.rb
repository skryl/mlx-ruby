# frozen_string_literal: true

require_relative "test_helper"

class Phase114PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_114_contract
    sched = MLX::Optimizers.exponential_decay(1.0, 0.5)
    assert_in_delta 0.125, sched.call(3), 1e-8
    assert_respond_to MLX::Optimizers, :linear_schedule
  end
end
