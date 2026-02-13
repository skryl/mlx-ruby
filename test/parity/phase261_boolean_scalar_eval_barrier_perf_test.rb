# frozen_string_literal: true

require_relative "test_helper"

class Phase261BooleanScalarEvalBarrierPerfTest < Minitest::Test
  def test_core_allclose_does_not_force_explicit_eval
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_allclose\(.*?^}\n/m]
    refute_nil segment
    refute_match(/out\.eval\(\);/, segment)
  end

  def test_core_array_equal_does_not_force_explicit_eval
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_array_equal\(.*?^}\n/m]
    refute_nil segment
    refute_match(/out\.eval\(\);/, segment)
  end
end
