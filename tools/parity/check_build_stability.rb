#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require "time"

RUBY_ROOT = Pathname.new(File.expand_path("..", __dir__)).freeze
EXTCONF = RUBY_ROOT.join("ext", "mlx", "extconf.rb").freeze
OUT_FILE = RUBY_ROOT.join("tools", "parity", "reports", "build_stability.json").freeze

source = File.read(EXTCONF)

checks = {
  "gguf_disabled" => source.include?("-DMLX_BUILD_GGUF=OFF"),
  "safetensors_disabled" => source.include?("-DMLX_BUILD_SAFETENSORS=OFF"),
  "configure_retry_present" => source.include?("initial CMake configure failed"),
  "retry_cleans_build_root" => source.include?("FileUtils.rm_rf(build_root)")
}

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "checks" => checks
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
