# frozen_string_literal: true

require_relative "test_helper"

class Phase271FunctionReleaseGvlPlumbingPerfTest < Minitest::Test
  def test_function_wrap_helpers_accept_release_gvl_flag
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))

    wrap_vector = source[/static VALUE function_wrap_vector\(.*?^}\n/m]
    refute_nil wrap_vector
    assert_match(/bool release_gvl = false/, wrap_vector)
    assert_match(/wrapper->release_gvl = release_gvl;/, wrap_vector)

    wrap_args_kwargs = source[/static VALUE function_wrap_args_kwargs\(.*?^}\n/m]
    refute_nil wrap_args_kwargs
    assert_match(/bool release_gvl = false/, wrap_args_kwargs)
    assert_match(/wrapper->release_gvl = release_gvl;/, wrap_args_kwargs)
  end

  def test_native_only_wrappers_enable_release_gvl
    source = File.read(File.join(RUBY_ROOT, "ext", "mlx", "native.cpp"))

    compile_segment = source[/static VALUE core_compile\(.*?^}\n/m]
    refute_nil compile_segment
    assert_match(/function_wrap_vector\(std::move\(compiled\), refs, true\)/, compile_segment)

    import_segment = source[/static VALUE core_import_function\(.*?^}\n/m]
    refute_nil import_segment
    assert_match(/function_wrap_args_kwargs\(std::move\(wrapped\), refs, false, true\)/, import_segment)
  end
end
