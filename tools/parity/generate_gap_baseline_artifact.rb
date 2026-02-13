#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "time"

REPO_ROOT = Pathname.new(File.expand_path("../..", __dir__)).freeze
TOOLS_ROOT = REPO_ROOT.join("tools").freeze
PARITY_ROOT = TOOLS_ROOT.join("parity", "reports").freeze

TOOLS = [
  TOOLS_ROOT.join("parity", "generate_api_inventory.rb"),
  TOOLS_ROOT.join("parity", "generate_package_inventory.rb")
].freeze

API_FILE = PARITY_ROOT.join("api_inventory.json").freeze
PACKAGE_FILE = PARITY_ROOT.join("package_inventory.json").freeze
OUT_FILE = PARITY_ROOT.join("gap_baseline.json").freeze

TOOLS.each do |tool|
  stdout, stderr, status = Open3.capture3("ruby", tool.to_s)
  next if status.success?

  abort <<~MSG
    failed to run #{tool}
    stdout:
    #{stdout}
    stderr:
    #{stderr}
  MSG
end

api = JSON.parse(File.read(API_FILE))
package = JSON.parse(File.read(PACKAGE_FILE))

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "method_level_gap" => {
    "python_modules" => api.fetch("python").fetch("top_level_modules", []),
    "ruby_modules" => api.fetch("ruby").fetch("top_level_modules", []),
    "python_core_singleton_methods" => api.fetch("python").fetch("core_singleton_methods", []),
    "ruby_core_singleton_methods" => api.fetch("ruby").fetch("core_singleton_methods", []),
    "python_array_instance_methods" => api.fetch("python").fetch("array_instance_methods", []),
    "ruby_array_instance_methods" => api.fetch("ruby").fetch("array_instance_methods", []),
    "diff" => api.fetch("diff")
  },
  "module_level_coverage" => {
    "python_files" => package.fetch("python").fetch("files", []),
    "ruby_files" => package.fetch("ruby").fetch("files", []),
    "python_exports" => package.fetch("python").fetch("exports", {}),
    "ruby_exports" => package.fetch("ruby").fetch("exports", {}),
    "diff" => package.fetch("diff")
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
