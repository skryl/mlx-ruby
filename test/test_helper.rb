# frozen_string_literal: true

require "json"
require "timeout"
require "minitest/autorun"
require "open3"
require "rbconfig"
require "tmpdir"

RUBY_ROOT = File.expand_path("..", __dir__)
REPO_ROOT = File.expand_path("..", RUBY_ROOT)

module TestSupport
  module_function

  def build_native_extension!
    return if @native_built

    ext_dir = File.join(RUBY_ROOT, "ext", "mlx")
    makefile_path = File.join(ext_dir, "Makefile")
    bundle_path = File.join(ext_dir, "native.#{RbConfig::CONFIG.fetch('DLEXT', 'bundle')}")

    if native_build_required?(bundle_path)
      run_cmd!(%w[ruby extconf.rb], ext_dir) if makefile_stale?(makefile_path)
      run_cmd!(%w[make], ext_dir)
    end

    @native_built = true
  end

  def makefile_stale?(makefile_path)
    return true unless File.exist?(makefile_path)

    extconf_path = File.join(RUBY_ROOT, "ext", "mlx", "extconf.rb")
    File.mtime(makefile_path) < File.mtime(extconf_path)
  end

  def native_build_required?(bundle_path)
    return true if ENV["MLX_RUBY_FORCE_REBUILD"] == "1"
    return true unless File.exist?(bundle_path)

    File.mtime(bundle_path) < newest_native_input_mtime
  end

  def newest_native_input_mtime
    roots = [
      File.join(RUBY_ROOT, "ext", "mlx"),
      File.join(RUBY_ROOT, "lib", "mlx", "version.rb"),
      File.join(REPO_ROOT, "mlx"),
      File.join(REPO_ROOT, "cmake"),
      File.join(REPO_ROOT, "CMakeLists.txt")
    ]

    newest = Time.at(0)
    roots.each do |root|
      if File.file?(root)
        mtime = File.mtime(root)
        newest = mtime if mtime > newest
        next
      end
      next unless Dir.exist?(root)

      Dir.glob(File.join(root, "**", "*"), File::FNM_DOTMATCH).each do |path|
        next unless File.file?(path)
        next if path.include?("/build/")
        next if path.include?("/.git/")

        mtime = File.mtime(path)
        newest = mtime if mtime > newest
      end
    end

    newest
  end

  def python_sources_available?
    roots = [
      File.join(REPO_ROOT, "python", "src"),
      File.join(REPO_ROOT, "python", "mlx"),
      File.join(REPO_ROOT, "mlx", "python", "src"),
      File.join(REPO_ROOT, "mlx", "python", "mlx")
    ]

    roots.any? do |root|
      File.directory?(root) && !Dir.children(root).empty?
    end
  end

  def run_cmd!(cmd, chdir)
    stdout, stderr, status = Open3.capture3(*cmd, chdir: chdir)
    return if status.success?

    raise <<~MSG
      command failed: #{cmd.join(" ")}
      cwd: #{chdir}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end
end

raw_test_timeout = ENV.fetch("MLX_TEST_TIMEOUT", "10").to_i
TEST_TIMEOUT_SECONDS = raw_test_timeout.positive? ? raw_test_timeout : 10

module Minitest
  class Test
    alias_method :run_without_timeout, :run

    def run
      Timeout.timeout(TEST_TIMEOUT_SECONDS) { run_without_timeout }
    end
  end
end
