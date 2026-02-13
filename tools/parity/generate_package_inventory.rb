#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require "time"

RUBY_ROOT = Pathname.new(File.expand_path("..", __dir__)).freeze
REPO_ROOT = RUBY_ROOT.parent.freeze
PYTHON_MLX_ROOT = REPO_ROOT.join("python", "mlx").freeze
RUBY_MLX_ROOT = RUBY_ROOT.join("lib", "mlx").freeze
OUT_FILE = RUBY_ROOT.join("tools", "parity", "reports", "package_inventory.json").freeze

def list_python_files
  Dir.glob(PYTHON_MLX_ROOT.join("**", "*")).filter_map do |path|
    next unless File.file?(path)

    Pathname.new(path).relative_path_from(PYTHON_MLX_ROOT).to_s
  end.sort
end

def list_ruby_files
  ignored = %w[core.rb version.rb nn/base.rb].freeze
  Dir.glob(RUBY_MLX_ROOT.join("**", "*.rb")).filter_map do |path|
    next unless File.file?(path)

    rel = Pathname.new(path).relative_path_from(RUBY_MLX_ROOT).to_s
    next if ignored.include?(rel)

    rel
  end.compact.sort
end

def top_level_defs_and_classes(path)
  text = File.read(path)
  defs = text.scan(/^def\s+([A-Za-z_][A-Za-z0-9_]*)/).flatten
  classes = text.scan(/^class\s+([A-Za-z_][A-Za-z0-9_]*)/).flatten
  (defs + classes).reject { |name| name.start_with?("_") }.uniq.sort
end

def python_namespace_symbols(glob)
  Dir.glob(PYTHON_MLX_ROOT.join(glob)).flat_map do |path|
    next [] unless File.file?(path)
    next [] unless path.end_with?(".py")

    top_level_defs_and_classes(path)
  end.uniq.sort
end

def ruby_namespace_exports(mod)
  return [] if mod.nil?

  constants = mod.constants(false).map(&:to_s)
  methods = mod.singleton_methods(false).map(&:to_s)
  (constants + methods).uniq.sort
end

def normalize_name(name)
  name.to_s.gsub(/[^a-zA-Z0-9]/, "").downcase
end

def normalized_diff(left_names, right_names)
  left_map = left_names.each_with_object({}) { |name, out| out[normalize_name(name)] = name }
  right_keys = right_names.map { |name| normalize_name(name) }.uniq

  left_map.filter_map do |key, original|
    original unless right_keys.include?(key)
  end.sort
end

def map_python_file_to_ruby(rel)
  return nil unless rel.end_with?(".py")
  return nil if rel == "__main__.py"
  return nil if rel == "py.typed"
  return nil if rel == "_stub_patterns.txt"
  return nil if rel == "_reprlib_fix.py"

  mapped = rel.dup
  mapped = mapped.sub(%r{\A_distributed_utils/}, "distributed_utils/")
  mapped = "nn.rb" if mapped == "nn/__init__.py"
  mapped = "optimizers.rb" if mapped == "optimizers/__init__.py"
  mapped = "nn/layers.rb" if mapped == "nn/layers/__init__.py"
  mapped = mapped.sub(/\.py\z/, ".rb")
  mapped
end

python_files = list_python_files
ruby_files = list_ruby_files

$LOAD_PATH.unshift(RUBY_ROOT.join("lib").to_s)
require "mlx"

ruby_nn = ruby_namespace_exports(defined?(MLX::NN) ? MLX::NN : nil)
ruby_optimizers = ruby_namespace_exports(defined?(MLX::Optimizers) ? MLX::Optimizers : nil)
ruby_extension = ruby_namespace_exports(defined?(MLX::Extension) ? MLX::Extension : nil)
ruby_dist = ruby_namespace_exports(defined?(MLX::DistributedUtils) ? MLX::DistributedUtils : nil)

python_nn = python_namespace_symbols("nn/**/*.py")
python_optimizers = python_namespace_symbols("optimizers/**/*.py")
python_extension = python_namespace_symbols("extension.py")
python_dist = python_namespace_symbols("_distributed_utils/**/*.py")

python_nn |= %w[init losses utils]
python_optimizers |= %w[schedulers]

expected_ruby_files = python_files.filter_map { |rel| map_python_file_to_ruby(rel) }.uniq.sort

payload = {
  "generated_at" => Time.now.utc.iso8601,
  "python" => {
    "files" => python_files,
    "exports" => {
      "nn" => python_nn,
      "optimizers" => python_optimizers,
      "extension" => python_extension,
      "distributed_utils" => python_dist
    }
  },
  "ruby" => {
    "files" => ruby_files,
    "exports" => {
      "nn" => ruby_nn,
      "optimizers" => ruby_optimizers,
      "extension" => ruby_extension,
      "distributed_utils" => ruby_dist
    }
  },
  "diff" => {
    "files_missing_in_ruby" => (expected_ruby_files - ruby_files),
    "files_extra_in_ruby" => (ruby_files - expected_ruby_files),
    "nn_missing_in_ruby" => normalized_diff(python_nn, ruby_nn),
    "nn_extra_in_ruby" => normalized_diff(ruby_nn, python_nn),
    "optimizers_missing_in_ruby" => normalized_diff(python_optimizers, ruby_optimizers),
    "optimizers_extra_in_ruby" => normalized_diff(ruby_optimizers, python_optimizers),
    "extension_missing_in_ruby" => normalized_diff(python_extension, ruby_extension),
    "extension_extra_in_ruby" => normalized_diff(ruby_extension, python_extension),
    "distributed_utils_missing_in_ruby" => normalized_diff(python_dist, ruby_dist),
    "distributed_utils_extra_in_ruby" => normalized_diff(ruby_dist, python_dist)
  }
}

OUT_FILE.dirname.mkpath
OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
puts "wrote #{OUT_FILE}"
