# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase81BuildStabilityContractTest < Minitest::Test
  def test_build_stability_contract
    tool = File.join(RUBY_ROOT, "tools", "parity", "check_build_stability.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "build_stability.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "tool failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing build stability artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    checks = payload.fetch("checks")

    assert_equal true, checks.fetch("gguf_disabled")
    assert_equal true, checks.fetch("safetensors_disabled")
    assert_equal true, checks.fetch("configure_retry_present")
    assert_equal true, checks.fetch("retry_cleans_build_root")
  end
end
