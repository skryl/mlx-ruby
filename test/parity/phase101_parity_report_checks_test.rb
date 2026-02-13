# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase101ParityReportChecksTest < Minitest::Test
  def test_report_emits_array_and_overall_parity_checks
    tool = File.join(RUBY_ROOT, "tools", "parity", "generate_parity_report.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "report.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "report generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"

    payload = JSON.parse(File.read(out_file))
    checks = payload.fetch("checks")

    assert_equal true, checks.fetch("array_name_parity")
    assert_equal true, checks.fetch("overall_parity")
  end
end
