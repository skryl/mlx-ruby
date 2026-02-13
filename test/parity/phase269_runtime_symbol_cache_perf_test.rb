# frozen_string_literal: true

require_relative "test_helper"

class Phase269RuntimeSymbolCachePerfTest < Minitest::Test
  def test_runtime_hot_paths_avoid_direct_rb_intern_calls
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))

    ruby_call_segment = source[/static VALUE ruby_callable_call_protected\(.*?^}\n/m]
    refute_nil ruby_call_segment
    refute_match(/rb_intern\("call"\)/, ruby_call_segment)

    keyword_hash_segment = source[/static VALUE ruby_keyword_hash_from_arrays\(.*?^}\n/m]
    refute_nil keyword_hash_segment
    refute_match(/rb_intern\(/, keyword_hash_segment)

    hash_fetch_segment = source[/static VALUE hash_fetch_optional\(.*?^}\n/m]
    refute_nil hash_fetch_segment
    refute_match(/rb_intern\(/, hash_fetch_segment)
  end
end
