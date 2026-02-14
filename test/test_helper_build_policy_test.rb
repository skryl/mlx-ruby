# frozen_string_literal: true

require_relative "test_helper"

class TestHelperBuildPolicyTest < Minitest::Test
  def setup
    @singleton = class << TestSupport
      self
    end
    @restores = []
    TestSupport.instance_variable_set(:@native_built, nil)
  end

  def teardown
    @restores.reverse_each do |name, backup|
      if @singleton.instance_methods(false).include?(name)
        @singleton.remove_method(name)
      end
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end
    TestSupport.instance_variable_set(:@native_built, nil)
    ENV.delete("MLX_RUBY_FORCE_REBUILD")
  end

  def test_build_native_extension_reuses_loadable_bundle_without_sources
    calls = []
    stub_singleton_method(:native_build_required?) { |_bundle_path| true }
    stub_singleton_method(:native_bundle_loadable?) { |_bundle_path| true }
    stub_singleton_method(:native_rebuild_sources_available?) { false }
    stub_singleton_method(:makefile_stale?) { |_makefile_path| true }
    stub_singleton_method(:run_cmd!) do |cmd, _chdir|
      calls << cmd
    end

    TestSupport.build_native_extension!

    assert_equal [], calls
    assert_equal true, TestSupport.instance_variable_get(:@native_built)
  end

  def test_force_rebuild_still_runs_build_commands
    calls = []
    ENV["MLX_RUBY_FORCE_REBUILD"] = "1"

    stub_singleton_method(:native_build_required?) { |_bundle_path| true }
    stub_singleton_method(:native_bundle_loadable?) { |_bundle_path| true }
    stub_singleton_method(:native_rebuild_sources_available?) { false }
    stub_singleton_method(:makefile_stale?) { |_makefile_path| true }
    stub_singleton_method(:run_cmd!) do |cmd, _chdir|
      calls << cmd
    end

    TestSupport.build_native_extension!

    assert_includes calls, %w[ruby extconf.rb]
    assert_includes calls, %w[make]
  end

  def test_signature_mismatch_triggers_rebuild_when_sources_are_available
    calls = []
    writes = []

    stub_singleton_method(:native_build_required?) { |_bundle_path| false }
    stub_singleton_method(:native_rebuild_sources_available?) { true }
    stub_singleton_method(:native_build_signature_mismatch?) { |_signature_path| true }
    stub_singleton_method(:makefile_stale?) { |_makefile_path| false }
    stub_singleton_method(:run_cmd!) do |cmd, _chdir|
      calls << cmd
    end
    stub_singleton_method(:write_native_build_signature!) do |signature_path|
      writes << signature_path
    end

    TestSupport.build_native_extension!

    assert_includes calls, %w[ruby extconf.rb]
    assert_includes calls, %w[make]
    assert_equal 1, writes.length
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__dsl_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    if @singleton.instance_methods(false).include?(name)
      @singleton.remove_method(name)
    end
    @singleton.define_method(name, &block)
  end
end
