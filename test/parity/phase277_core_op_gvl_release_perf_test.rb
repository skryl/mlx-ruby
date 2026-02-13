# frozen_string_literal: true

require_relative "test_helper"

class Phase277CoreOpGvlReleasePerfTest < Minitest::Test
  TARGET_FUNCTIONS = %w[core_add core_subtract core_multiply core_divide core_matmul].freeze

  def test_high_traffic_core_ops_use_shared_no_gvl_array_helper
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    assert_match(/static mx::array call_mx_array_without_gvl\(/, source)

    TARGET_FUNCTIONS.each do |fn_name|
      segment = source[/static VALUE #{Regexp.escape(fn_name)}\(.*?^}\n/m]
      refute_nil segment, "missing function segment for #{fn_name}"
      assert_match(/call_mx_array_without_gvl\(/, segment, "#{fn_name} should execute compute without GVL")
    end
  end
end
