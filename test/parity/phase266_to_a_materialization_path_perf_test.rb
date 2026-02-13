# frozen_string_literal: true

require_relative "test_helper"

class Phase266ToAMaterializationPathPerfTest < Minitest::Test
  def test_array_to_a_avoids_unconditional_pre_eval_for_non_scalar_paths
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE array_to_a\(.*?^}\n/m]
    refute_nil segment

    pre_branch = segment[/\A.*?if \(wrapper->array.ndim\(\) == 0\)/m]
    refute_nil pre_branch
    refute_match(/wrapper->array\.eval\(\);/, pre_branch)
    assert_match(/if \(wrapper->array.ndim\(\) == 0\) \{\n\s*wrapper->array\.eval\(\);/m, segment)
  end
end
