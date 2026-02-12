# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase103ModuleNameParityContractTest < Minitest::Test
  def test_top_level_module_name_parity_has_no_missing_entries
    tool = File.join(RUBY_ROOT, "tools", "generate_api_inventory.rb")
    out_file = File.join(RUBY_ROOT, "parity", "api_inventory.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    payload = JSON.parse(File.read(out_file))

    diff = payload.fetch("diff")
    missing = diff.fetch("top_level_missing_in_ruby")
    assert_equal [], missing, "missing top-level modules: #{missing.inspect}"
  end
end
