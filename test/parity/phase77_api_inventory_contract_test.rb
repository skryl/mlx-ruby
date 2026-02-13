# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase77ApiInventoryContractTest < Minitest::Test
  def test_api_inventory_generator_writes_contract_artifact
    tool = File.join(RUBY_ROOT, "tools", "parity", "generate_api_inventory.rb")
    out_file = File.join(RUBY_ROOT, "tools", "parity", "reports", "api_inventory.json")

    stdout, stderr, status = Open3.capture3("ruby", tool)
    assert status.success?, "generator failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert File.exist?(out_file), "missing inventory artifact at #{out_file}"

    payload = JSON.parse(File.read(out_file))
    assert payload.key?("generated_at")
    assert payload.key?("python")
    assert payload.key?("ruby")
    assert payload.key?("diff")

    python = payload.fetch("python")
    ruby = payload.fetch("ruby")
    diff = payload.fetch("diff")

    assert_includes python.fetch("top_level_modules"), "mlx.nn"
    assert_includes python.fetch("top_level_modules"), "mlx.optimizers"
    assert_includes python.fetch("top_level_modules"), "mlx.utils"
    assert_operator python.fetch("core_singleton_methods").length, :>, 200
    assert_operator ruby.fetch("core_singleton_methods").length, :>, 200

    assert diff.key?("core_missing_in_ruby")
    assert_equal [], diff.fetch("core_missing_in_ruby")
  end
end
