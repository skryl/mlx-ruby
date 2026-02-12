# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase208FunctionalHarnessDistributedUtilsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def test_harness_distributed_utils_section
    tool = File.join(RUBY_ROOT, "tools", "generate_functional_golden_report.rb")
    out_file = File.join(RUBY_ROOT, "parity", "functional_golden_report.json")
    _stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "functional golden harness failed\nstderr:\n#{stderr}"

    payload = JSON.parse(File.read(out_file))
    section = payload.fetch("sections").fetch("distributed_utils")
    assert_equal true, section.fetch("all_pass")
  end
end
