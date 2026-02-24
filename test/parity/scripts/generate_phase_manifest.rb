#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require "yaml"
require "fileutils"
require_relative "phase_manifest_builder"

options = {
  repo_root: File.expand_path("../../..", __dir__),
  output: nil
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby test/parity/scripts/generate_phase_manifest.rb [--repo-root PATH] [--output PATH]"
  opts.on("--repo-root PATH", "Repository root") { |value| options[:repo_root] = value }
  opts.on("--output PATH", "Manifest output path") { |value| options[:output] = value }
end

parser.parse!(ARGV)

repo_root = File.expand_path(options[:repo_root])
default_output = File.join(repo_root, "test", "parity", "manifest.yml")
output_path = File.expand_path(options[:output] || default_output)

payload = PhaseManifestBuilder.build(repo_root: repo_root)

FileUtils.mkdir_p(File.dirname(output_path))
File.write(output_path, YAML.dump(payload))

puts "Wrote phase manifest to #{output_path}"
