# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase82ParityReportArtifactTest < Minitest::Test
  def test_consolidated_parity_report_is_generated
    tool = File.join(RUBY_ROOT, "tools", "parity", "generate_parity_report.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "report.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing report artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("summary")
    assert payload.key?("checks")
    assert payload.key?("gaps")

    summary = payload.fetch("summary")
    assert summary.fetch("core_missing_in_ruby").is_a?(Integer)
    assert summary.fetch("array_missing_in_ruby").is_a?(Integer)

    checks = payload.fetch("checks")
    assert_equal true, checks.fetch("build_stability_contract")
  end
end
