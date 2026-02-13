# frozen_string_literal: true

require_relative "test_helper"

class Phase270ToAVectorFastPathPerfTest < Minitest::Test
  def test_array_to_a_has_dedicated_1d_fast_path
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE array_to_a\(.*?^}\n/m]
    refute_nil segment

    assert_match(/if \(wrapper->array\.ndim\(\) == 1\)/, segment)
    assert_match(/if \(wrapper->array\.ndim\(\) == 1\) \{\n\s*wrapper->array\.eval\(\);/m, segment)
  end
end
