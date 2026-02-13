# frozen_string_literal: true

require_relative "test_helper"

class Phase267GvlReleaseEntrypointsPerfTest < Minitest::Test
  def test_function_call_has_no_gvl_path
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE function_call\(.*?^}\n/m]
    refute_nil segment
    assert_match(/rb_thread_call_without_gvl\(/, segment)
  end

  def test_core_eval_has_no_gvl_path
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_eval\(.*?^}\n/m]
    refute_nil segment
    assert_match(/rb_thread_call_without_gvl\(/, segment)
  end

  def test_core_async_eval_has_no_gvl_path
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_async_eval\(.*?^}\n/m]
    refute_nil segment
    assert_match(/rb_thread_call_without_gvl\(/, segment)
  end
end
