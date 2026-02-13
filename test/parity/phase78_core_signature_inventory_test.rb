# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase78CoreSignatureInventoryTest < Minitest::Test
  def test_core_signature_inventory_contract
    tool = File.join(RUBY_ROOT, "tools", "parity", "generate_core_signature_inventory.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "core_signature_inventory.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing signature artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("python")
    assert payload.key?("ruby")
    assert payload.key?("diff")

    ruby_methods = payload.fetch("ruby").fetch("singleton_methods")
    python_methods = payload.fetch("python").fetch("singleton_methods")
    diff = payload.fetch("diff")

    %w[compile grad value_and_grad export_function import_function stream].each do |name|
      assert ruby_methods.key?(name), "missing ruby method #{name}"
      assert python_methods.key?(name), "missing python method #{name}"
    end

    assert diff.key?("missing_from_ruby")
    assert_equal [], diff.fetch("missing_from_ruby")
  end
end
