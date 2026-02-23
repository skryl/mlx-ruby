# frozen_string_literal: true

require "stringio"
require_relative "../test_helper"
require_relative "../../tasks/web_assets_task"

class WebAssetsTaskTest < Minitest::Test
  def setup
    @singleton = class << WebAssetsTask
      self
    end
    @restores = []
  end

  def teardown
    @restores.reverse_each do |name, backup|
      @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end
  end

  def test_run_emits_verbose_progress_for_each_script
    scripts = ["/tmp/export_a.rb", "/tmp/export_b.rb"]
    calls = []
    timestamps = [10.0, 11.5, 12.0, 13.5, 14.0, 16.0]

    stub_singleton_method(:script_paths) { scripts }
    stub_singleton_method(:system) do |ruby_bin, script|
      calls << [ruby_bin, script]
      true
    end
    stub_singleton_method(:monotonic_now) { timestamps.shift || 16.0 }

    out = StringIO.new
    WebAssetsTask.run!(ruby_bin: "/custom/ruby", out: out)
    output = out.string

    assert_equal [
      ["/custom/ruby", "/tmp/export_a.rb"],
      ["/custom/ruby", "/tmp/export_b.rb"]
    ], calls
    assert_includes output, "[web:assets] Resolving export scripts..."
    assert_includes output, "[web:assets] Found 2 script(s) to run."
    assert_includes output, "[web:assets]   1. /tmp/export_a.rb"
    assert_includes output, "[web:assets]   2. /tmp/export_b.rb"
    assert_includes output, "[web:assets] Ruby executable: /custom/ruby"
    assert_includes output, "[web:assets] (1/2) Starting export_a.rb"
    assert_includes output, "[web:assets] (1/2) Completed export_a.rb in 0.50s"
    assert_includes output, "[web:assets] (2/2) Starting export_b.rb"
    assert_includes output, "[web:assets] (2/2) Completed export_b.rb in 0.50s"
    assert_includes output, "[web:assets] Finished web asset export in 6.00s"
  end

  def test_run_emits_failure_step_and_raises
    scripts = ["/tmp/export_fail.rb"]
    timestamps = [30.0, 31.0, 33.5]

    stub_singleton_method(:script_paths) { scripts }
    stub_singleton_method(:system) do |ruby_bin, script|
      !(ruby_bin == "/custom/ruby" && script == "/tmp/export_fail.rb")
    end
    stub_singleton_method(:monotonic_now) { timestamps.shift || 33.5 }

    out = StringIO.new

    error = assert_raises(RuntimeError) do
      WebAssetsTask.run!(ruby_bin: "/custom/ruby", out: out)
    end

    assert_equal "Web assets export failed: /tmp/export_fail.rb", error.message
    assert_includes out.string, "[web:assets] (1/1) Failed export_fail.rb after 2.50s"
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__web_assets_task_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
    @singleton.define_method(name, &block)
  end
end
