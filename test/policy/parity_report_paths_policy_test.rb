# frozen_string_literal: true

require "open3"
require_relative "../support/test_helper"

class TestParityReportPathsPolicyTest < Minitest::Test
  def test_parity_scripts_write_generated_reports_under_test_tmp
    legacy_output = File.join(RUBY_ROOT, "test", "parity", "reports", "build_stability.json")
    generated_output = TestSupport.parity_generated_report_path("build_stability.json")
    script = File.join(RUBY_ROOT, "test", "parity", "scripts", "check_build_stability.rb")

    FileUtils.rm_f(legacy_output)
    FileUtils.rm_f(generated_output)

    _stdout, stderr, status = Open3.capture3("ruby", script)
    assert status.success?, "check_build_stability failed\nstderr:\n#{stderr}"

    assert File.exist?(generated_output), "expected generated report at #{generated_output}"
    refute File.exist?(legacy_output), "expected legacy report path to remain unused"
  end

  def test_phase0_contract_lives_under_snapshots
    snapshot_contract = TestSupport.parity_snapshot_path("phase0_contract.json")
    legacy_contract = File.join(RUBY_ROOT, "test", "parity", "reports", "phase0_contract.json")

    assert File.exist?(snapshot_contract), "missing snapshot contract at #{snapshot_contract}"
    refute File.exist?(legacy_contract), "legacy contract path should be empty: #{legacy_contract}"
  end
end
