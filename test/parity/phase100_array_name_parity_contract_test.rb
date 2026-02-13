# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase100ArrayNameParityContractTest < Minitest::Test
  def test_python_array_names_have_no_missing_entries_in_ruby_inventory
    tool = File.join(RUBY_ROOT, "tools", "parity", "generate_api_inventory.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "api_inventory.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    payload = JSON.parse(File.read(out_file))

    missing = payload.fetch("diff").fetch("array_missing_in_ruby")
    assert_equal [], missing, "missing array names: #{missing.inspect}"
  end
end
