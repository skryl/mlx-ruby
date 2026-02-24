# frozen_string_literal: true

require "stringio"
require_relative "../support/test_helper"
require_relative "../../tasks/web_assets_task"

class WebAssetsTaskTest < Minitest::Test
  SCRIPT_NAMES_BY_TARGET = {
    "gpt2" => "export_gpt2_assets.rb",
    "stable_diffusion" => "export_stable_diffusion_assets.rb",
    "nanogpt" => "export_nanogpt_assets.rb"
  }.freeze

  def setup
    @singleton = class << WebAssetsTask
      self
    end
    @restores = []
    @env_restores = {}
  end

  def teardown
    @restores.reverse_each do |name, backup|
      @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end
    @env_restores.each do |key, value|
      if value.nil?
        ENV.delete(key)
      else
        ENV[key] = value
      end
    end
  end

  def test_run_emits_verbose_progress_for_each_script
    scripts = ["/tmp/export_a.rb", "/tmp/export_b.rb"]
    calls = []
    timestamps = [10.0, 11.5, 12.0, 13.5, 14.0, 16.0]

    stub_singleton_method(:script_paths) { |targets: nil| scripts }
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
    assert_includes output, "[web:assets] Targets: gpt2, stable_diffusion, nanogpt"
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

    stub_singleton_method(:script_paths) { |targets: nil| scripts }
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

  def test_selected_targets_defaults_to_all_targets
    with_env("WEB_ASSETS_TARGETS", nil) do
      assert_equal SCRIPT_NAMES_BY_TARGET.keys, WebAssetsTask.selected_targets
    end
  end

  def test_selected_targets_respects_env_subset_in_canonical_order
    with_env("WEB_ASSETS_TARGETS", "nanogpt,gpt2") do
      assert_equal %w[gpt2 nanogpt], WebAssetsTask.selected_targets
    end
  end

  def test_selected_targets_rejects_unknown_target
    with_env("WEB_ASSETS_TARGETS", "gpt2,unknown_demo") do
      error = assert_raises(ArgumentError) { WebAssetsTask.selected_targets }
      assert_includes error.message, "Unknown WEB_ASSETS_TARGETS values: unknown_demo"
    end
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__web_assets_task_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
    @singleton.define_method(name, &block)
  end

  def with_env(key, value)
    @env_restores[key] = ENV[key] unless @env_restores.key?(key)
    if value.nil?
      ENV.delete(key)
    else
      ENV[key] = value
    end
    yield
  ensure
    original = @env_restores[key]
    if original.nil?
      ENV.delete(key)
    else
      ENV[key] = original
    end
  end
end
