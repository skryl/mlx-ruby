# frozen_string_literal: true

require_relative "test_helper"

class Phase268NestedArrayIngestFastPathPerfTest < Minitest::Test
  def test_tensor_array_from_ruby_has_typed_float_fast_path
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static mx::array tensor_array_from_ruby\(.*?^}\n/m]
    refute_nil segment
    assert_match(/std::vector<float>/, segment)
  end
end
