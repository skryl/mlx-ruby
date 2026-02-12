#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require "time"

RUBY_ROOT = Pathname.new(File.expand_path("..", __dir__)).freeze
REPO_ROOT = RUBY_ROOT.parent.freeze
PYTHON_ROOT = REPO_ROOT.join("python").freeze
PYTHON_MLX_ROOT = PYTHON_ROOT.join("mlx").freeze
PYTHON_SRC_ROOT = PYTHON_ROOT.join("src").freeze
RUBY_LIB_ROOT = RUBY_ROOT.join("lib").freeze
RUBY_NATIVE_CPP = RUBY_ROOT.join("ext", "mlx", "native.cpp").freeze
OUT_FILE = RUBY_ROOT.join("parity", "api_inventory.json").freeze

def scan_regex(path, regex)
  path = Pathname.new(path)
  return [] unless path.exist?

  File.read(path).scan(regex).flatten.uniq.sort
end

def python_top_level_modules
  modules = ["mlx", "mlx.core"]

  Dir.children(PYTHON_MLX_ROOT).sort.each do |entry|
    full = PYTHON_MLX_ROOT.join(entry)
    if full.directory? && full.join("__init__.py").exist?
      modules << "mlx.#{entry}"
      next
    end

    next unless full.file?
    next unless entry.end_with?(".py")
    next if entry.start_with?("_")

    stem = File.basename(entry, ".py")
    next if stem == "__main__"

    modules << "mlx.#{stem}"
  end

  modules.uniq.sort
end

def ruby_top_level_modules
  modules = ["MLX"]

  Dir.glob(RUBY_LIB_ROOT.join("mlx", "*.rb")).sort.each do |path|
    stem = File.basename(path, ".rb")
    const = stem.split("_").map(&:capitalize).join
    modules << "MLX::#{const}"
  end

  modules.uniq.sort
end

def ruby_modules_as_python_names(modules)
  modules.map do |name|
    parts = name.split("::")
    next name.downcase if parts.empty?

    parts.map(&:downcase).join(".")
  end
end

def python_core_singleton_methods
  Dir.glob(PYTHON_SRC_ROOT.join("*.cpp")).flat_map do |path|
    scan_regex(path, /m\.def\(\s*(?:\n\s*)*"([^"]+)"/m)
  end.uniq.sort
end

def ruby_core_singleton_methods
  scan_regex(
    RUBY_NATIVE_CPP,
    /rb_define_singleton_method\(mCore,\s*"([^"]+)"/
  )
end

def ruby_array_instance_methods
  lib_path = RUBY_LIB_ROOT.to_s
  $LOAD_PATH.unshift(lib_path) unless $LOAD_PATH.include?(lib_path)
  require "mlx"

  return [] unless defined?(MLX::Core::Array)

  MLX::Core::Array.instance_methods(false).map(&:to_s).uniq.sort
rescue LoadError, StandardError
  scan_regex(
    RUBY_NATIVE_CPP,
    /rb_define_method\(cArray,\s*"([^"]+)"/
  )
ensure
  $LOAD_PATH.delete(lib_path) if defined?(lib_path)
end

def python_array_instance_methods
  path = PYTHON_SRC_ROOT.join("array.cpp")
  return [] unless path.exist?

  text = File.read(path)
  marker = "nb::class_<mx::array>("
  start = text.index(marker)
  return [] if start.nil?

  segment = text[start..]
  patterns = [
    /\.def\(\s*"([^"]+)"/,
    /\.def_prop_ro\(\s*"([^"]+)"/,
    /\.def_prop_rw\(\s*"([^"]+)"/,
    /\.def_ro\(\s*"([^"]+)"/,
    /\.def_rw\(\s*"([^"]+)"/
  ]

  patterns.flat_map { |regex| segment.scan(regex).flatten }.uniq.sort
end

python_core = python_core_singleton_methods
ruby_core = ruby_core_singleton_methods
python_array = python_array_instance_methods
ruby_array = ruby_array_instance_methods
python_modules = python_top_level_modules
ruby_modules = ruby_top_level_modules
ruby_modules_as_python = ruby_modules_as_python_names(ruby_modules)

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "python" => {
    "top_level_modules" => python_modules,
    "core_singleton_methods" => python_core,
    "array_instance_methods" => python_array
  },
  "ruby" => {
    "top_level_modules" => ruby_modules,
    "core_singleton_methods" => ruby_core,
    "array_instance_methods" => ruby_array
  },
  "diff" => {
    "top_level_missing_in_ruby" => (python_modules - ruby_modules_as_python),
    "top_level_extra_in_ruby" => (ruby_modules_as_python - python_modules),
    "core_missing_in_ruby" => (python_core - ruby_core),
    "core_extra_in_ruby" => (ruby_core - python_core),
    "array_missing_in_ruby" => (python_array - ruby_array),
    "array_extra_in_ruby" => (ruby_array - python_array)
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
