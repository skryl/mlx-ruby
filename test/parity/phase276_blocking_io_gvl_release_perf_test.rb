# frozen_string_literal: true

require_relative "test_helper"

class Phase276BlockingIoGvlReleasePerfTest < Minitest::Test
  def test_core_synchronize_releases_gvl
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_synchronize\(.*?^}\n/m]
    refute_nil segment
    assert_match(/rb_thread_call_without_gvl\(/, segment)
  end

  def test_core_save_releases_gvl
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_save\(.*?^}\n/m]
    refute_nil segment
    assert_match(/rb_thread_call_without_gvl\(/, segment)
  end

  def test_core_load_npy_branch_releases_gvl
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))
    segment = source[/static VALUE core_load\(.*?^}\n/m]
    refute_nil segment
    assert_match(/format_v == "npy".*rb_thread_call_without_gvl\(/m, segment)
  end
end
