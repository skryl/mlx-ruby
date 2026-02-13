# frozen_string_literal: true

require "open3"
require_relative "test_helper"

class Phase147PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_147_contract
    ruby_value = MLX::Optimizers.exponential_decay(1.0, 0.5).call(4)
    stdout, _stderr, status = Open3.capture3("python3", "-c", "print((1.0) * (0.5 ** 4))")
    assert status.success?
    assert_in_delta stdout.to_f, ruby_value, 1e-8
  end
end
