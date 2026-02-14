#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "time"

REPO_ROOT = Pathname.new(File.expand_path("../../..", __dir__)).freeze
SCRIPT_ROOT = REPO_ROOT.join("test", "parity", "scripts").freeze
PARITY_ROOT = REPO_ROOT.join("test", "parity", "reports").freeze
INVENTORY_TOOL = SCRIPT_ROOT.join("generate_package_inventory.rb").freeze
INVENTORY_FILE = PARITY_ROOT.join("package_inventory.json").freeze
OUT_FILE = PARITY_ROOT.join("package_report.json").freeze

stdout, stderr, status = Open3.capture3("ruby", INVENTORY_TOOL.to_s)
unless status.success?
  abort <<~MSG
    failed to run #{INVENTORY_TOOL}
    stdout:
    #{stdout}
    stderr:
    #{stderr}
  MSG
end

inventory = JSON.parse(File.read(INVENTORY_FILE))
diff = inventory.fetch("diff")

summary = {
  "files_missing_in_ruby" => diff.fetch("files_missing_in_ruby").length,
  "files_extra_in_ruby" => diff.fetch("files_extra_in_ruby").length,
  "nn_missing_in_ruby" => diff.fetch("nn_missing_in_ruby").length,
  "nn_extra_in_ruby" => diff.fetch("nn_extra_in_ruby").length,
  "optimizers_missing_in_ruby" => diff.fetch("optimizers_missing_in_ruby").length,
  "optimizers_extra_in_ruby" => diff.fetch("optimizers_extra_in_ruby").length,
  "extension_missing_in_ruby" => diff.fetch("extension_missing_in_ruby").length,
  "extension_extra_in_ruby" => diff.fetch("extension_extra_in_ruby").length,
  "distributed_utils_missing_in_ruby" => diff.fetch("distributed_utils_missing_in_ruby").length,
  "distributed_utils_extra_in_ruby" => diff.fetch("distributed_utils_extra_in_ruby").length
}

checks = {
  "file_surface_parity" => summary["files_missing_in_ruby"].zero?,
  "nn_name_parity" => summary["nn_missing_in_ruby"].zero?,
  "optimizers_name_parity" => summary["optimizers_missing_in_ruby"].zero?,
  "extension_name_parity" => summary["extension_missing_in_ruby"].zero?,
  "distributed_utils_name_parity" => summary["distributed_utils_missing_in_ruby"].zero?
}
checks["overall_package_parity"] = checks.values.all?

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "summary" => summary,
  "checks" => checks,
  "gaps" => {
    "files_missing_in_ruby" => diff.fetch("files_missing_in_ruby"),
    "nn_missing_in_ruby" => diff.fetch("nn_missing_in_ruby"),
    "optimizers_missing_in_ruby" => diff.fetch("optimizers_missing_in_ruby"),
    "extension_missing_in_ruby" => diff.fetch("extension_missing_in_ruby"),
    "distributed_utils_missing_in_ruby" => diff.fetch("distributed_utils_missing_in_ruby")
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
