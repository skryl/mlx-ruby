# frozen_string_literal: true

require_relative "test_helper"

class Phase1BootstrapTest < Minitest::Test
  GEMSPEC_PATH = File.join(RUBY_ROOT, "mlx.gemspec")

  def test_gemspec_exists_and_declares_native_extension
    assert File.file?(GEMSPEC_PATH), "missing gemspec at #{GEMSPEC_PATH}"

    spec = Gem::Specification.load(GEMSPEC_PATH)
    refute_nil spec, "failed to load gemspec"

    assert_equal "mlx", spec.name
    assert_includes spec.files, "lib/mlx.rb"
    assert_includes spec.extensions, "ext/mlx/extconf.rb"
  end

  def test_library_loads_without_native_extension
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"

    assert defined?(MLX)
    assert_match(/\A\d+\.\d+\.\d+(?:\.\d+)?\z/, MLX::VERSION)
    assert_respond_to MLX, :native_available?
    assert_includes [true, false], MLX.native_available?
  ensure
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_extension_build_files_exist
    extconf = File.join(RUBY_ROOT, "ext", "mlx", "extconf.rb")
    cmake_lists = File.join(RUBY_ROOT, "ext", "mlx", "CMakeLists.txt")

    assert File.file?(extconf), "missing extconf.rb"
    assert File.file?(cmake_lists), "missing CMakeLists.txt"
  end
end
