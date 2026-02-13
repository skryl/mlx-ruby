# frozen_string_literal: true

require_relative "test_helper"

class Phase265NativeSymbolIdCachePerfTest < Minitest::Test
  TARGETS = [
    "device_type_from_value",
    "dtype_from_symbol",
    "symbol_is_dtype",
    "dtype_to_symbol",
    "category_from_symbol",
    "category_to_symbol",
    "dtype_or_category_from_value"
  ].freeze

  def test_targeted_conversion_functions_avoid_direct_rb_intern_calls
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))

    TARGETS.each do |fn_name|
      segment = source[/static [^\n]*\b#{Regexp.escape(fn_name)}\(.*?^}\n/m]
      refute_nil segment, "missing function segment for #{fn_name}"
      refute_match(/rb_intern\(/, segment, "#{fn_name} should use cached IDs instead of direct rb_intern")
    end
  end
end
