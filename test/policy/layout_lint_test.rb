# frozen_string_literal: true

require "yaml"
require_relative "../support/test_helper"

class TestLayoutLintTest < Minitest::Test
  MANIFEST_PATH = File.join(RUBY_ROOT, "test", "reports", "test_layout_manifest.txt").freeze
  PARITY_MANIFEST_PATH = File.join(RUBY_ROOT, "test", "parity", "manifest.yml").freeze

  def test_no_legacy_onnx_or_web_test_directories
    refute Dir.exist?(File.join(RUBY_ROOT, "test", "onnx")), "legacy test/onnx directory should not exist"
    refute Dir.exist?(File.join(RUBY_ROOT, "test", "web")), "legacy test/web directory should not exist"
  end

  def test_no_top_level_test_ruby_files
    top_level_ruby_files = Dir.glob(File.join(RUBY_ROOT, "test", "*.rb")).sort
    assert_empty top_level_ruby_files, "top-level test/*.rb files should be moved under suite/support directories"
  end

  def test_test_layout_manifest_matches_current_test_inventory
    assert File.exist?(MANIFEST_PATH), "missing layout manifest: #{MANIFEST_PATH}"
    expected = test_file_inventory
    actual = File.readlines(MANIFEST_PATH, chomp: true).reject(&:empty?).sort
    assert_equal expected, actual
  end

  def test_parity_phase_files_are_fully_mapped_in_manifest
    assert File.exist?(PARITY_MANIFEST_PATH), "missing parity manifest: #{PARITY_MANIFEST_PATH}"
    payload = YAML.safe_load(File.binread(PARITY_MANIFEST_PATH), permitted_classes: [Time], aliases: false)
    phase_entries = payload.fetch("phases")
    mapped_files = phase_entries.values.map { |entry| entry.fetch("file") }.sort
    actual_phase_files = Dir.glob(File.join(RUBY_ROOT, "test", "parity", "phase*_test.rb"))
      .map { |path| relative(path) }
      .sort

    assert_equal actual_phase_files, mapped_files
  end

  def test_generated_parity_json_reports_are_not_written_to_source_reports_dir
    source_reports_dir = File.join(RUBY_ROOT, "test", "parity", "reports")
    json_files = Dir.glob(File.join(source_reports_dir, "*.json"))
    assert_empty json_files, "source reports dir should not contain generated JSON files"
  end

  private

  def test_file_inventory
    Dir.glob(File.join(RUBY_ROOT, "test", "**", "*_test.rb"))
      .sort
      .map { |path| relative(path) }
  end

  def relative(path)
    path.delete_prefix("#{RUBY_ROOT}/")
  end
end
