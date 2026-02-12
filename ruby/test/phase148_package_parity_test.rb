# frozen_string_literal: true

require "open3"
require "json"
require_relative "test_helper"

class Phase148PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_148_contract
    tool = File.join(RUBY_ROOT, "tools", "generate_package_parity_report.rb")
    out_file = File.join(RUBY_ROOT, "parity", "package_report.json")
    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "package report failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    payload = JSON.parse(File.read(out_file))
    summary = payload.fetch("summary")
    %w[files_missing_in_ruby files_extra_in_ruby nn_missing_in_ruby nn_extra_in_ruby optimizers_missing_in_ruby optimizers_extra_in_ruby extension_missing_in_ruby extension_extra_in_ruby distributed_utils_missing_in_ruby distributed_utils_extra_in_ruby].each { |k| assert_equal 0, summary.fetch(k) }
    assert_equal true, payload.fetch("checks").fetch("overall_package_parity")
  end
end
