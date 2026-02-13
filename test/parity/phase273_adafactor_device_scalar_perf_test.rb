# frozen_string_literal: true

require_relative "test_helper"

class Phase273AdafactorDeviceScalarPerfTest < Minitest::Test
  def test_adafactor_avoids_item_to_f_scalar_host_sync
    source = File.read(File.join(RUBY_ROOT, "lib", "mlx", "optimizers", "optimizers.rb"))
    segment = source[/class Adafactor < Optimizer.*?^    end\n    class Muon/m]
    refute_nil segment

    refute_match(/item\.to_f/, segment)
    refute_match(/def scalar\(/, segment)
  end
end
