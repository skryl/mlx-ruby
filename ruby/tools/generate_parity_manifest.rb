#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "fileutils"
require "optparse"
require "time"

class ManifestGenerator
  def initialize(repo_root)
    @repo_root = File.expand_path(repo_root)
  end

  def generate
    {
      "metadata" => {
        "generated_at" => Time.now.utc.iso8601,
        "repo_root" => @repo_root,
        "generator" => "ruby/tools/generate_parity_manifest.rb"
      },
      "python_binding" => python_binding,
      "python_package" => python_package,
      "python_tests" => python_tests
    }
  end

  private

  def python_binding
    src_files = Dir.glob(File.join(@repo_root, "python", "src", "*.cpp")).sort

    def_count_by_file = {}
    m_def_count_by_file = {}
    function_names = []
    class_names = []
    enum_names = []
    submodule_names = []
    constants = []

    src_files.each do |path|
      content = File.read(path)
      file = File.basename(path)

      def_count_by_file[file] = content.scan(/\.def\(/).length
      m_def_count_by_file[file] = content.scan(/m\.def\(/).length

      function_names.concat(content.scan(/m\.def\(\s*(?:\n\s*)*"([^"]+)"/m).flatten)
      class_names.concat(content.scan(/class_<[^>]+>\s*\(\s*[^,]+,\s*"([^"]+)"/m).flatten)
      enum_names.concat(content.scan(/enum_<[^>]+>\s*\(\s*[^,]+,\s*"([^"]+)"/m).flatten)
      submodule_names.concat(content.scan(/def_submodule\(\s*"([^"]+)"/m).flatten)
      constants.concat(content.scan(/m\.attr\(\s*"([^"]+)"/m).flatten)
    end

    {
      "files" => src_files.map { |path| relative(path) },
      "defs" => {
        "total" => def_count_by_file.values.sum,
        "m_def_total" => m_def_count_by_file.values.sum,
        "by_file" => def_count_by_file,
        "m_def_by_file" => m_def_count_by_file
      },
      "symbols" => {
        "functions" => function_names.uniq.sort,
        "classes" => class_names.uniq.sort,
        "enums" => enum_names.uniq.sort,
        "submodules" => submodule_names.uniq.sort,
        "constants" => constants.uniq.sort
      }
    }
  end

  def python_package
    nn_files = Dir.glob(File.join(@repo_root, "python", "mlx", "nn", "**", "*.py")).sort
    optimizer_files = Dir.glob(File.join(@repo_root, "python", "mlx", "optimizers", "*.py")).sort
    utils_files = [
      File.join(@repo_root, "python", "mlx", "utils.py"),
      File.join(@repo_root, "python", "mlx", "nn", "utils.py")
    ]

    {
      "nn" => python_code_stats(nn_files),
      "optimizers" => python_code_stats(optimizer_files),
      "utils" => python_code_stats(utils_files)
    }
  end

  def python_tests
    test_files = Dir.glob(File.join(@repo_root, "python", "tests", "test_*.py")).sort
    by_file = {}

    test_files.each do |path|
      count = File.foreach(path).count { |line| line.match?(/^\s*def\s+test_/) }
      by_file[File.basename(path)] = count
    end

    {
      "files" => test_files.map { |path| relative(path) },
      "total_test_cases" => by_file.values.sum,
      "by_file" => by_file
    }
  end

  def python_code_stats(paths)
    class_count = 0
    function_count = 0

    paths.each do |path|
      next unless File.file?(path)

      File.foreach(path) do |line|
        class_count += 1 if line.match?(/^\s*class\s+/)
        function_count += 1 if line.match?(/^\s*def\s+/)
      end
    end

    {
      "files" => paths.select { |path| File.file?(path) }.map { |path| relative(path) },
      "class_count" => class_count,
      "function_count" => function_count
    }
  end

  def relative(path)
    path.delete_prefix(@repo_root + File::SEPARATOR)
  end
end

options = {
  repo_root: nil,
  output: nil
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby/tools/generate_parity_manifest.rb --repo-root PATH --output PATH"

  opts.on("--repo-root PATH", "Repository root") do |value|
    options[:repo_root] = value
  end

  opts.on("--output PATH", "Output JSON path") do |value|
    options[:output] = value
  end
end

parser.parse!(ARGV)

if options[:repo_root].nil? || options[:output].nil?
  warn parser.banner
  exit 1
end

manifest = ManifestGenerator.new(options[:repo_root]).generate
output_path = File.expand_path(options[:output])
FileUtils.mkdir_p(File.dirname(output_path))
File.write(output_path, JSON.pretty_generate(manifest) + "\n")

puts "Wrote parity manifest to #{output_path}"
