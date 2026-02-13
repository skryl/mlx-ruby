#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require "time"

REPO_ROOT = Pathname.new(File.expand_path("../..", __dir__)).freeze
TOOLS_ROOT = REPO_ROOT.join("tools").freeze
PARITY_ROOT = TOOLS_ROOT.join("parity").freeze
PYTHON_SRC_ROOT = REPO_ROOT.join("python", "src").freeze
RUBY_NATIVE_CPP = REPO_ROOT.join("ext", "mlx", "native.cpp").freeze
OUT_FILE = PARITY_ROOT.join("reports", "core_signature_inventory.json").freeze

ruby_source = File.read(RUBY_NATIVE_CPP)

ruby_methods = {}
ruby_source.scan(
  /rb_define_singleton_method\(\s*mCore,\s*"([^"]+)",\s*RUBY_METHOD_FUNC\(([^)]+)\),\s*(-?\d+)\s*\);/m
) do |name, c_func, arity_text|
  arity = arity_text.to_i
  ruby_methods[name] = {
    "c_func" => c_func,
    "arity" => arity,
    "variadic" => arity.negative?
  }
end

python_methods = {}
if PYTHON_SRC_ROOT.directory?
  Dir.glob(PYTHON_SRC_ROOT.join("*.cpp")).sort.each do |path|
    source = File.read(path)

    source.scan(/m\.def\(\s*(?:\n\s*)*"([^"]+)"/m).flatten.each do |name|
      python_methods[name] ||= {
        "sources" => [],
        "signatures" => []
      }
      python_methods[name]["sources"] << File.basename(path)
    end

    source.scan(/m\.def\(\s*"([^"]+)".*?nb::sig\("([^"]+)"\)/m).each do |name, sig|
      python_methods[name] ||= {
        "sources" => [],
        "signatures" => []
      }
      python_methods[name]["signatures"] << sig
    end
  end
end

python_methods.each_value do |entry|
  entry["sources"] = entry["sources"].uniq.sort
  entry["signatures"] = entry["signatures"].uniq.sort
  entry["has_signature"] = !entry["signatures"].empty?
end

ruby_names = ruby_methods.keys.sort
python_names = python_methods.keys.sort

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "python" => {
    "singleton_methods" => python_methods,
    "count" => python_names.length
  },
  "ruby" => {
    "singleton_methods" => ruby_methods,
    "count" => ruby_names.length
  },
  "diff" => {
    "missing_from_ruby" => (python_names - ruby_names),
    "missing_from_python" => (ruby_names - python_names),
    "ruby_fixed_arity_without_python_signature" => ruby_names.filter_map do |name|
      ruby = ruby_methods[name]
      python = python_methods[name]
      next if ruby.nil? || python.nil?
      next if ruby["variadic"]
      next if python["has_signature"]

      name
    end
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
