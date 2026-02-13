#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "optparse"

require "pathname"

DEFAULT_CONTRACT = Pathname.new(File.expand_path("../..", __dir__))
  .join("tools", "parity", "reports", "phase0_contract.json")
  .to_s

options = {
  manifest: nil,
  contract: DEFAULT_CONTRACT
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby/tools/parity/check_parity_manifest.rb --manifest PATH [--contract PATH]"

  opts.on("--manifest PATH", "Path to generated manifest") do |value|
    options[:manifest] = value
  end

  opts.on("--contract PATH", "Path to parity contract JSON") do |value|
    options[:contract] = value
  end
end

parser.parse!(ARGV)

if options[:manifest].nil?
  warn parser.banner
  exit 1
end

manifest = JSON.parse(File.read(File.expand_path(options[:manifest])))
contract = JSON.parse(File.read(File.expand_path(options[:contract])))

errors = []

if manifest.dig("python_binding", "defs", "total").to_i < contract["min_binding_defs_total"].to_i
  errors << "python_binding.defs.total below contract threshold"
end

if manifest.dig("python_binding", "defs", "m_def_total").to_i < contract["min_binding_m_def_total"].to_i
  errors << "python_binding.defs.m_def_total below contract threshold"
end

functions = manifest.dig("python_binding", "symbols", "functions") || []
(contract["required_functions"] || []).each do |name|
  errors << "missing required function: #{name}" unless functions.include?(name)
end

if manifest.dig("python_package", "nn", "class_count").to_i < contract["min_nn_class_count"].to_i
  errors << "python_package.nn.class_count below contract threshold"
end

if manifest.dig("python_tests", "total_test_cases").to_i < contract["min_python_tests_total"].to_i
  errors << "python_tests.total_test_cases below contract threshold"
end

if manifest.dig("python_tests", "by_file", "test_ops.py").to_i < contract["min_test_ops_cases"].to_i
  errors << "python_tests.by_file.test_ops.py below contract threshold"
end

if errors.empty?
  puts "Parity contract OK"
  exit 0
end

warn "Parity contract FAILED"
errors.each { |line| warn "- #{line}" }
exit 1
