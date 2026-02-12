#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "time"

RUBY_ROOT = Pathname.new(File.expand_path("..", __dir__)).freeze
PARITY_ROOT = RUBY_ROOT.join("parity").freeze

TOOLS = [
  RUBY_ROOT.join("tools", "generate_api_inventory.rb"),
  RUBY_ROOT.join("tools", "generate_core_signature_inventory.rb"),
  RUBY_ROOT.join("tools", "check_build_stability.rb")
].freeze

API_FILE = PARITY_ROOT.join("api_inventory.json").freeze
SIG_FILE = PARITY_ROOT.join("core_signature_inventory.json").freeze
BUILD_FILE = PARITY_ROOT.join("build_stability.json").freeze
OUT_FILE = PARITY_ROOT.join("report.json").freeze

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
sig = JSON.parse(File.read(SIG_FILE))
build = JSON.parse(File.read(BUILD_FILE))

build_ok = build.fetch("checks").values.all?

summary = {
  "core_missing_in_ruby" => api.fetch("diff").fetch("core_missing_in_ruby").length,
  "core_extra_in_ruby" => api.fetch("diff").fetch("core_extra_in_ruby").length,
  "array_missing_in_ruby" => api.fetch("diff").fetch("array_missing_in_ruby").length,
  "array_extra_in_ruby" => api.fetch("diff").fetch("array_extra_in_ruby").length,
  "signature_missing_from_ruby" => sig.fetch("diff").fetch("missing_from_ruby").length
}

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "summary" => summary,
  "checks" => {
    "build_stability_contract" => build_ok,
    "core_name_parity" => summary["core_missing_in_ruby"].zero?,
    "core_signature_name_parity" => summary["signature_missing_from_ruby"].zero?
  },
  "gaps" => {
    "core_missing_in_ruby" => api.fetch("diff").fetch("core_missing_in_ruby"),
    "array_missing_in_ruby" => api.fetch("diff").fetch("array_missing_in_ruby"),
    "signature_missing_from_ruby" => sig.fetch("diff").fetch("missing_from_ruby")
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
