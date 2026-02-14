# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase107PackageReportContractTest < Minitest::Test
  def test_package_parity_report_is_generated
    tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_package_parity_report.rb")
    out_file = File.join(RUBY_ROOT, "test", "parity", "reports", "package_report.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "report generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing report artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("summary")
    assert payload.key?("checks")
    assert payload.key?("gaps")

    checks = payload.fetch("checks")
    assert checks.key?("nn_name_parity")
    assert checks.key?("optimizers_name_parity")
    assert checks.key?("extension_name_parity")
    assert checks.key?("distributed_utils_name_parity")
    assert checks.key?("overall_package_parity")
  end
end
