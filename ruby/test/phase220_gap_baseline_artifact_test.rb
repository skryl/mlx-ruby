# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase220GapBaselineArtifactTest < Minitest::Test
  def test_gap_baseline_artifact_contains_method_and_module_coverage
    tool = File.join(RUBY_ROOT, "tools", "generate_gap_baseline_artifact.rb")
    out_file = File.join(RUBY_ROOT, "parity", "gap_baseline.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("method_level_gap")
    assert payload.key?("module_level_coverage")

    method_gap = payload.fetch("method_level_gap")
    assert method_gap.fetch("python_core_singleton_methods").is_a?(Array)
    assert method_gap.fetch("ruby_core_singleton_methods").is_a?(Array)
    assert method_gap.fetch("diff").is_a?(Hash)

    module_coverage = payload.fetch("module_level_coverage")
    assert module_coverage.fetch("python_files").is_a?(Array)
    assert module_coverage.fetch("ruby_files").is_a?(Array)
    assert module_coverage.fetch("diff").is_a?(Hash)
  end
end
