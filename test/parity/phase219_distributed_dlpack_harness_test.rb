# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase219DistributedDlpackHarnessTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def test_distributed_and_dlpack_golden_harness
    tool = File.join(RUBY_ROOT, "test", "parity", "scripts", "generate_functional_golden_report.rb")
    out_file = File.join(RUBY_ROOT, "test", "parity", "reports", "functional_golden_report.json")
    _stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "functional golden harness failed\nstderr:\n#{stderr}"

    payload = JSON.parse(File.read(out_file))
    section = payload.fetch("sections").fetch("distributed_utils")
    assert_equal true, section.fetch("all_pass")

    checks = section.fetch("checks")
    names = checks.map { |c| c.fetch("name") }
    %w[dlpack_roundtrip config_main_dispatch_ethernet launch_main_print_python].each do |name|
      assert_includes names, name
      assert_equal true, checks.find { |entry| entry["name"] == name }["pass"]
    end
  end
end
