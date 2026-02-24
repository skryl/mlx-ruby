# frozen_string_literal: true

require "fileutils"
require "open3"
require "rbconfig"
require "rake"
require "rake/file_list"
require "timeout"
require "tmpdir"

class MlxTestTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze

  def self.strict_mode?
    ENV.fetch("MLX_STRICT_TESTS", "0") == "1"
  end

  def self.strict_test_timeout_seconds
    value = ENV.fetch("MLX_TEST_TIMEOUT", "10").to_i
    value.positive? ? value : 10
  end

  def self.configure_base_test_task(task)
    task.libs << "test"
    task.pattern = "test/**/*_test.rb"
    task.warning = true
  end

  def self.parse_test_devices_arg(raw_devices)
    return [] if raw_devices.nil?

    raw_devices
      .to_s
      .split(",")
      .map(&:strip)
      .reject(&:empty?)
      .map(&:downcase)
  end

  def self.test_file_list
    Rake::FileList["test/**/*_test.rb"]
  end

  def self.selected_test_files(pattern: ENV["TEST"]&.strip)
    return test_file_list.to_a if pattern.nil? || pattern.empty?

    Rake::FileList[pattern].to_a
  end

  def self.run_test_suite_for_devices(raw_devices, include_slow: false)
    devices = parse_test_devices_arg(raw_devices)
    devices = %w[cpu gpu] if devices.empty?

    devices.each do |device|
      puts "==> Running test suite with DEVICE=#{device}"
      run_test_suite_for_device(device, include_slow: include_slow)
    end
  end

  def self.run_test_suite_for_device(device = nil, include_slow: false)
    if device.nil?
      run_base_test_task(include_slow: include_slow)
      return
    end

    with_forced_test_device(device) do
      run_base_test_task(include_slow: include_slow)
    end
  end

  def self.run_base_test_task(include_slow: false)
    with_filtered_gem_platform_warnings do
      with_include_slow_tests(include_slow) do
        Rake::Task[:test_base].reenable
        Rake::Task[:test_base].invoke
      end
    end
  end

  def self.with_include_slow_tests(include_slow)
    had_value = ENV.key?("MLX_TEST_INCLUDE_SLOW")
    previous_value = ENV["MLX_TEST_INCLUDE_SLOW"]

    if include_slow
      ENV["MLX_TEST_INCLUDE_SLOW"] = "1"
    else
      ENV.delete("MLX_TEST_INCLUDE_SLOW")
    end

    yield
  ensure
    if had_value
      ENV["MLX_TEST_INCLUDE_SLOW"] = previous_value
    else
      ENV.delete("MLX_TEST_INCLUDE_SLOW")
    end
  end

  def self.run_strict_test_suite!
    files = selected_test_files
    failures = []

    files.each do |file|
      print "."
      $stdout.flush
      success = run_test_file_with_timeout(file)
      failures << file unless success
    end

    puts
    puts "Ran #{files.length} tests in strict mode (#{strict_test_timeout_seconds}s timeout)."

    return unless failures.any?

    warn
    warn "The following files failed or timed out:"
    failures.each { |file| warn "  - #{file}" }
    raise "Strict test run failed."
  end

  def self.run_installed_gem_test_suite!
    Dir.mktmpdir("mlx-ruby-gem-test-") do |tmp_dir|
      gem_home = File.join(tmp_dir, "gems")
      gem_file = File.join(tmp_dir, "mlx.gem")
      run_root = File.join(tmp_dir, "ruby")
      cache_home = File.join(tmp_dir, "cache")
      clang_module_cache = File.join(cache_home, "clang", "ModuleCache")

      FileUtils.mkdir_p(gem_home)
      FileUtils.mkdir_p(cache_home)
      FileUtils.mkdir_p(clang_module_cache)

      gem_env = unbundled_env(
        "GEM_HOME" => gem_home,
        "GEM_PATH" => gem_home,
        "HOME" => tmp_dir,
        "XDG_CACHE_HOME" => cache_home,
        "CLANG_MODULE_CACHE_PATH" => clang_module_cache
      )

      install_gemspec_development_dependencies!(File.join(REPO_ROOT, "mlx.gemspec"), env: gem_env, chdir: REPO_ROOT)
      run_command!(["gem", "build", "mlx.gemspec", "--output", gem_file], env: gem_env, chdir: REPO_ROOT)
      run_command!(["gem", "install", "--local", "--no-document", gem_file], env: gem_env, chdir: REPO_ROOT)

      spec_info_script = <<~'RUBY'
        spec = Gem::Specification.find_by_name("mlx")
        puts spec.full_gem_path
        puts spec.extension_dir
      RUBY

      spec_info = capture_command_output!(
        [RbConfig.ruby, "-rrubygems", "-e", spec_info_script],
        env: gem_env,
        chdir: REPO_ROOT
      ).lines.map(&:strip)

      full_gem_path = spec_info[0]
      extension_dir = spec_info[1]
      if full_gem_path.nil? || full_gem_path.empty? || extension_dir.nil? || extension_dir.empty?
        raise "Failed to resolve installed mlx gem paths from isolated GEM_HOME at #{gem_home}."
      end

      native_bundle_path = installed_native_bundle_path(extension_dir)
      raise "Could not find installed native extension under #{extension_dir}." if native_bundle_path.nil?

      prepare_installed_gem_test_root(run_root, full_gem_path, native_bundle_path)

      test_env = gem_env.merge(
        "MLX_TEST_RUBY_ROOT" => run_root,
        "MLX_TEST_REPO_ROOT" => REPO_ROOT,
        "MLX_TEST_SKIP_NATIVE_BUILD" => "1"
      )

      load_check_script = <<~'RUBY'
        require "mlx"
        mlx_feature = $LOADED_FEATURES.find { |path| path.end_with?("/mlx.rb") }
        native_feature = $LOADED_FEATURES.find do |path|
          path.match?(%r{/mlx/native\.(?:bundle|so)\z}) ||
            path.match?(%r{/native\.(?:bundle|so)\z})
        end
        puts "MLX_FEATURE=#{mlx_feature}"
        puts "NATIVE_FEATURE=#{native_feature}"
      RUBY

      load_check_output = capture_command_output!(
        [RbConfig.ruby, "-Itest", "-e", load_check_script],
        env: test_env,
        chdir: REPO_ROOT
      )
      mlx_feature = load_check_output[/^MLX_FEATURE=(.+)$/, 1]
      native_feature = load_check_output[/^NATIVE_FEATURE=(.+)$/, 1]
      if mlx_feature.nil? || native_feature.nil?
        raise "Failed installed-gem load check. Output:\n#{load_check_output}"
      end

      repo_ext_native_prefix = File.join(REPO_ROOT, "ext", "mlx", "native")
      if native_feature.start_with?(repo_ext_native_prefix)
        raise "Installed-gem load check failed: native extension resolved to repo ext path #{native_feature}."
      end

      files = selected_test_files
      failures = []

      files.each do |file|
        print "."
        $stdout.flush
        success = run_test_file_with_timeout(
          file,
          command: [RbConfig.ruby, "-Itest", file],
          chdir: REPO_ROOT,
          env: test_env
        )
        failures << file unless success
      end

      puts
      puts "Ran #{files.length} tests against installed gem from #{full_gem_path}."

      next if failures.empty?

      warn
      warn "The following files failed or timed out:"
      failures.each { |file| warn "  - #{file}" }
      raise "Installed gem test run failed."
    end
  end

  def self.run_test_file_with_timeout(
    file,
    timeout: strict_test_timeout_seconds,
    command: nil,
    chdir: REPO_ROOT,
    env: {}
  )
    command ||= ["bundle", "exec", "ruby", "-Itest", file]
    pid = Process.spawn(env, *command, chdir: chdir)

    begin
      _, status = Timeout.timeout(timeout) { Process.wait2(pid) }
      return status.success?
    rescue Timeout::Error
      warn "x timeout after #{timeout}s: #{file}"
      begin
        Process.kill("TERM", pid)
      rescue Errno::ESRCH, Errno::EINVAL, Errno::EPERM
        return false
      end

      sleep 0.2
      begin
        Process.kill("KILL", pid)
      rescue Errno::ESRCH, Errno::EINVAL, Errno::EPERM
        # ignore
      end

      Process.wait(pid) rescue nil
      false
    end
  ensure
    begin
      Process.kill("KILL", pid) if pid
    rescue Errno::ESRCH, Errno::EINVAL, Errno::EPERM
      # ignore if process already exited
    end
  end

  def self.unbundled_env(overrides = {})
    env = {}

    ENV.each_key do |key|
      env[key] = nil if key.start_with?("BUNDLE_") || key.start_with?("BUNDLER_")
    end

    rubyopt_tokens = ENV.fetch("RUBYOPT", "").split
    rubyopt_tokens.reject! { |token| token.include?("bundler/setup") }
    env["RUBYOPT"] = rubyopt_tokens.empty? ? nil : rubyopt_tokens.join(" ")
    env["RUBYLIB"] = nil if ENV.key?("RUBYLIB")
    env["RUBYGEMS_GEMDEPS"] = nil if ENV.key?("RUBYGEMS_GEMDEPS")

    overrides.each do |key, value|
      env[key] = value
    end

    env
  end

  def self.capture_command_output!(command, env: {}, chdir: REPO_ROOT)
    stdout, stderr, status = Open3.capture3(env, *command, chdir: chdir)
    return stdout if status.success?

    raise <<~MSG
      command failed: #{command.join(" ")}
      cwd: #{chdir}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end

  def self.run_command!(command, env: {}, chdir: REPO_ROOT)
    success = system(env, *command, chdir: chdir)
    return if success

    raise "command failed: #{command.join(' ')} (cwd: #{chdir})"
  end

  def self.gem_installed?(name, requirement, env: {}, chdir: REPO_ROOT)
    command = ["gem", "list", "-i", name]
    normalized_requirement = requirement.to_s.strip
    unless normalized_requirement.empty? || normalized_requirement == ">= 0"
      command += ["-v", normalized_requirement]
    end

    system(env, *command, chdir: chdir, out: File::NULL, err: File::NULL)
  end

  def self.gemspec_development_dependencies(gemspec_path, env: {}, chdir: REPO_ROOT)
    script = <<~'RUBY'
      spec = Gem::Specification.load(ARGV.fetch(0))
      abort("Failed to load gemspec at #{ARGV.fetch(0)}") unless spec
      spec.development_dependencies.each do |dep|
        puts "#{dep.name}\t#{dep.requirement}"
      end
    RUBY

    output = capture_command_output!(
      [RbConfig.ruby, "-rrubygems", "-e", script, gemspec_path],
      env: env,
      chdir: chdir
    )

    output
      .lines
      .map(&:strip)
      .reject(&:empty?)
      .map do |line|
        name, requirement = line.split("\t", 2)
        [name, (requirement || ">= 0")]
      end
  end

  def self.install_gemspec_development_dependencies!(gemspec_path, env: {}, chdir: REPO_ROOT)
    gemspec_development_dependencies(gemspec_path, env: env, chdir: chdir).each do |name, requirement|
      next if gem_installed?(name, requirement, env: env, chdir: chdir)

      command = ["gem", "install", "--no-document", name]
      normalized_requirement = requirement.to_s.strip
      unless normalized_requirement.empty? || normalized_requirement == ">= 0"
        command += ["-v", normalized_requirement]
      end
      run_command!(command, env: env, chdir: chdir)
    end
  end

  def self.installed_native_bundle_path(extension_dir)
    dlext = RbConfig::CONFIG.fetch("DLEXT", "bundle")
    Dir.glob(File.join(extension_dir, "**", "native.#{dlext}"))
      .sort
      .find { |path| File.file?(path) }
  end

  def self.prepare_installed_gem_test_root(run_root, full_gem_path, native_bundle_path)
    dlext = RbConfig::CONFIG.fetch("DLEXT", "bundle")
    run_ext_mlx_dir = File.join(run_root, "ext", "mlx")
    installed_ext_mlx_dir = File.join(full_gem_path, "ext", "mlx")

    FileUtils.mkdir_p(run_root)
    FileUtils.mkdir_p(run_ext_mlx_dir)
    FileUtils.mkdir_p(File.join(run_root, "tmp"))

    FileUtils.cp(File.join(REPO_ROOT, "mlx.gemspec"), File.join(run_root, "mlx.gemspec"))
    FileUtils.ln_s(File.join(REPO_ROOT, "test"), File.join(run_root, "test"))
    FileUtils.ln_s(File.join(full_gem_path, "lib"), File.join(run_root, "lib"))

    if Dir.exist?(installed_ext_mlx_dir)
      Dir.glob(File.join(installed_ext_mlx_dir, "*"), File::FNM_DOTMATCH).each do |path|
        next if [".", ".."].include?(File.basename(path))

        FileUtils.cp_r(path, run_ext_mlx_dir)
      end
    end

    native_bundle_link = File.join(run_ext_mlx_dir, "native.#{dlext}")
    FileUtils.rm_f(native_bundle_link)
    FileUtils.ln_s(native_bundle_path, native_bundle_link)
  end

  def self.with_forced_test_device(device)
    normalized = device.to_s.downcase
    unless %w[cpu gpu metal].include?(normalized)
      raise ArgumentError, "Unsupported test device: #{device.inspect}. Use :cpu, :gpu, or :metal."
    end

    previous_device = ENV["DEVICE"]
    had_mlx_default_device = ENV.key?("MLX_DEFAULT_DEVICE")
    previous_mlx_default_device = ENV["MLX_DEFAULT_DEVICE"]

    ENV["DEVICE"] = normalized
    ENV.delete("MLX_DEFAULT_DEVICE")

    yield
  ensure
    if previous_device.nil?
      ENV.delete("DEVICE")
    else
      ENV["DEVICE"] = previous_device
    end

    if had_mlx_default_device
      ENV["MLX_DEFAULT_DEVICE"] = previous_mlx_default_device
    else
      ENV.delete("MLX_DEFAULT_DEVICE")
    end
  end

  def self.with_filtered_gem_platform_warnings
    read_io, write_io = IO.pipe
    original_stderr = STDERR.dup
    stderr_reader = Thread.new do
      begin
        read_io.each_line do |line|
          next if gem_platform_warning_line?(line)

          original_stderr.write(line)
        end
      ensure
        read_io.close unless read_io.closed?
      end
    end

    STDERR.reopen(write_io)
    $stderr = STDERR
    write_io.close unless write_io.closed?

    yield
  ensure
    STDERR.reopen(original_stderr) if original_stderr && !original_stderr.closed?
    $stderr = STDERR
    original_stderr.close if original_stderr && !original_stderr.closed?
    stderr_reader.join if stderr_reader
  end

  def self.gem_platform_warning_line?(line)
    return true if line.include?("warning: already initialized constant Gem::Platform::")
    return false unless line.include?("warning: previous definition of")

    line.include?("/rubygems/platform.rb:")
  end
end
