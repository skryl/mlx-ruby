# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase106PackageInventoryContractTest < Minitest::Test
  def test_package_inventory_generator_writes_contract_artifact
    tool = File.join(RUBY_ROOT, "tools", "generate_package_inventory.rb")
    out_file = File.join(RUBY_ROOT, "parity", "package_inventory.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing package inventory artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("python")
    assert payload.key?("ruby")
    assert payload.key?("diff")

    python = payload.fetch("python")
    ruby = payload.fetch("ruby")
    diff = payload.fetch("diff")

    assert python.key?("files")
    assert python.key?("exports")
    assert ruby.key?("files")
    assert ruby.key?("exports")
    assert diff.key?("nn_missing_in_ruby")
    assert diff.key?("optimizers_missing_in_ruby")
    assert diff.key?("extension_missing_in_ruby")
    assert diff.key?("distributed_utils_missing_in_ruby")
  end
end
