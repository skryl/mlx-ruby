# frozen_string_literal: true

require_relative "../support/test_helper"

class TestHelperSupportSplitTest < Minitest::Test
  def test_native_build_methods_live_in_support_module
    source = TestSupport.method(:build_native_extension!).source_location&.first
    assert source
    assert_includes source, "/test/support/native_build.rb"
  end

  def test_slow_test_methods_live_in_support_module
    source = TestSupport.method(:include_slow_tests?).source_location&.first
    assert source
    assert_includes source, "/test/support/slow_tests.rb"
  end

  def test_export_helper_methods_live_in_support_module
    source = TestSupport.method(:export_graph_ir_to_target).source_location&.first
    assert source
    assert_includes source, "/test/support/export_helpers.rb"
  end

  def test_tmp_helper_methods_live_in_support_module
    source = TestSupport.method(:test_tmp_dir).source_location&.first
    assert source
    assert_includes source, "/test/support/tmp_helpers.rb"
  end
end
