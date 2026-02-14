# frozen_string_literal: true

require "json"
require "timeout"
require "minitest/autorun"
require "open3"
require "rbconfig"
require "tmpdir"
require "fileutils"

RUBY_ROOT = File.expand_path("..", __dir__)
REPO_ROOT = File.expand_path("..", RUBY_ROOT)

module TestSupport
  module_function

  def build_native_extension!
    return if @native_built

    ext_dir = File.join(RUBY_ROOT, "ext", "mlx")
    makefile_path = File.join(ext_dir, "Makefile")
    bundle_path = File.join(ext_dir, "native.#{RbConfig::CONFIG.fetch('DLEXT', 'bundle')}")
    signature_path = native_build_signature_path(ext_dir)
    signature_mismatch = native_build_signature_mismatch?(signature_path)

    if reuse_loadable_native_bundle_without_sources?(bundle_path)
      @native_built = true
      return
    end

    if native_build_required?(bundle_path) || signature_mismatch
      run_cmd!(%w[ruby extconf.rb], ext_dir) if makefile_stale?(makefile_path) || signature_mismatch
      run_cmd!(%w[make], ext_dir)
      write_native_build_signature!(signature_path)
    end

    @native_built = true
  end

  def native_build_signature_path(ext_dir)
    File.join(ext_dir, ".native_build_signature")
  end

  def native_build_signature_mismatch?(signature_path)
    return false unless native_rebuild_sources_available?

    expected = current_native_build_signature
    return true unless File.exist?(signature_path)

    File.read(signature_path).strip != expected
  end

  def write_native_build_signature!(signature_path)
    return unless native_rebuild_sources_available?

    File.write(signature_path, "#{current_native_build_signature}\n")
  end

  def current_native_build_signature
    [
      RUBY_VERSION,
      RbConfig::CONFIG.fetch("arch", ""),
      newest_native_input_mtime.to_i,
      git_head_revision(RUBY_ROOT),
      git_head_revision(REPO_ROOT)
    ].join("|")
  end

  def git_head_revision(path)
    return nil unless Dir.exist?(path)

    stdout, _stderr, status = Open3.capture3("git", "-C", path, "rev-parse", "HEAD")
    return nil unless status.success?

    stdout.strip
  rescue Errno::ENOENT
    nil
  end

  def reuse_loadable_native_bundle_without_sources?(bundle_path)
    return false if ENV["MLX_RUBY_FORCE_REBUILD"] == "1"
    return false unless native_build_required?(bundle_path)
    return false unless native_bundle_loadable?(bundle_path)
    return false if native_rebuild_sources_available?

    true
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

  def native_bundle_loadable?(bundle_path)
    return false unless File.exist?(bundle_path)

    begin
      require bundle_path
    rescue LoadError
      begin
        require File.join(RUBY_ROOT, "ext", "mlx", "native")
      rescue LoadError
        return false
      end
    end

    defined?(MLX::Native) && MLX::Native.respond_to?(:loaded?) && MLX::Native.loaded?
  end

  def native_rebuild_sources_available?
    File.exist?(File.join(REPO_ROOT, "mlx", "CMakeLists.txt"))
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

  def test_tmp_dir
    @test_tmp_dir ||= begin
      path = File.join(RUBY_ROOT, "test", "tmp")
      FileUtils.mkdir_p(path)
      path
    end
  end

  def mktmpdir(prefix = "mlx-ruby-")
    return Dir.mktmpdir(prefix, test_tmp_dir) unless block_given?

    Dir.mktmpdir(prefix, test_tmp_dir) do |dir|
      yield dir
    end
  end
end

begin
  TestSupport.build_native_extension!
rescue StandardError
  nil
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
