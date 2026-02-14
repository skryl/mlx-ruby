# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase104ParityReportModuleChecksTest < Minitest::Test
  def test_report_tracks_top_level_module_parity
    tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_parity_report.rb")
    out_file = File.join(RUBY_ROOT, "test", "parity", "reports", "report.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "report generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"

    payload = JSON.parse(File.read(out_file))
    summary = payload.fetch("summary")
    checks = payload.fetch("checks")

    assert_equal 0, summary.fetch("top_level_missing_in_ruby")
    assert_equal true, checks.fetch("top_level_module_parity")
    assert_equal true, checks.fetch("overall_parity")
  end
end
