# frozen_string_literal: true

require_relative "test_helper"

class Phase272NestedIngestReservePerfTest < Minitest::Test
  def test_infer_shape_and_flatten_typed_reserves_capacity_for_known_subtrees
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/template <typename T>\nstatic void infer_shape_and_flatten_typed\(.*?^}\n/m]
    refute_nil segment

    assert_match(/flat\.reserve\(/, segment)
    assert_match(/shape\.size\(\) > depth \+ 1/, segment)
  end
end
