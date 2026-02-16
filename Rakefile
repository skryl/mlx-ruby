# frozen_string_literal: true

require "rake"
require "rake/clean"
require "rake/file_list"
require "rake/testtask"
require "rbconfig"
require "fileutils"
require "open3"
require "tmpdir"
require "timeout"

CLEAN.include(
  "ext/mlx/build",
  "ext/mlx/Makefile",
  "ext/mlx/native.o",
  "ext/mlx/native.bundle",
  "ext/mlx/native.bundle.dSYM"
)

def strict_test_timeout_seconds
  value = ENV.fetch("MLX_TEST_TIMEOUT", "10").to_i
  value.positive? ? value : 10
end

def gem_platform_warning_line?(line)
  return true if line.include?("warning: already initialized constant Gem::Platform::")
  return false unless line.include?("warning: previous definition of")

  line.include?("/rubygems/platform.rb:")
end

def with_filtered_gem_platform_warnings
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

def test_file_list
  Rake::FileList["test/**/*_test.rb"]
end

def selected_test_files
  pattern = ENV["TEST"]&.strip
  return test_file_list.to_a if pattern.nil? || pattern.empty?

  Rake::FileList[pattern].to_a
end

def run_test_file_with_timeout(
  file,
  timeout: strict_test_timeout_seconds,
  command: nil,
  chdir: __dir__,
  env: {}
)
  command ||= ["bundle", "exec", "ruby", "-Itest", file]
  pid = Process.spawn(env, *command, chdir: chdir)

  begin
    _, status = Timeout.timeout(timeout) { Process.wait2(pid) }
    return status.success?
  rescue Timeout::Error
    warn "✗ timeout after #{timeout}s: #{file}"
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

def unbundled_env(overrides = {})
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

def capture_command_output!(command, env: {}, chdir: __dir__)
  stdout, stderr, status = Open3.capture3(env, *command, chdir: chdir)
  return stdout if status.success?

  abort <<~MSG
    command failed: #{command.join(" ")}
    cwd: #{chdir}
    stdout:
    #{stdout}
    stderr:
    #{stderr}
  MSG
end

def run_command!(command, env: {}, chdir: __dir__)
  success = system(env, *command, chdir: chdir)
  return if success

  abort("command failed: #{command.join(" ")} (cwd: #{chdir})")
end

def gem_installed?(name, requirement, env: {}, chdir: __dir__)
  command = ["gem", "list", "-i", name]
  normalized_requirement = requirement.to_s.strip
  unless normalized_requirement.empty? || normalized_requirement == ">= 0"
    command += ["-v", normalized_requirement]
  end

  system(env, *command, chdir: chdir, out: File::NULL, err: File::NULL)
end

def gemspec_development_dependencies(gemspec_path, env: {}, chdir: __dir__)
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

def install_gemspec_development_dependencies!(gemspec_path, env: {}, chdir: __dir__)
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

def installed_native_bundle_path(extension_dir)
  dlext = RbConfig::CONFIG.fetch("DLEXT", "bundle")
  Dir.glob(File.join(extension_dir, "**", "native.#{dlext}"))
    .sort
    .find { |path| File.file?(path) }
end

def prepare_installed_gem_test_root(run_root, full_gem_path, native_bundle_path)
  dlext = RbConfig::CONFIG.fetch("DLEXT", "bundle")
  run_ext_mlx_dir = File.join(run_root, "ext", "mlx")
  installed_ext_mlx_dir = File.join(full_gem_path, "ext", "mlx")

  FileUtils.mkdir_p(run_root)
  FileUtils.mkdir_p(run_ext_mlx_dir)
  FileUtils.mkdir_p(File.join(run_root, "tmp"))

  FileUtils.cp(File.join(__dir__, "mlx.gemspec"), File.join(run_root, "mlx.gemspec"))
  FileUtils.ln_s(File.join(__dir__, "test"), File.join(run_root, "test"))
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

def run_strict_test_suite
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
  abort "Strict test run failed."
end

def with_forced_test_device(device)
  normalized = device.to_s.downcase
  unless %w[cpu gpu metal].include?(normalized)
    raise ArgumentError, "Unsupported test device: #{device.inspect}. Use :cpu, :gpu, or :metal."
  end

  previous_device = ENV["DEVICE"]
  had_mlx_default_device = ENV.key?("MLX_DEFAULT_DEVICE")
  previous_mlx_default_device = ENV["MLX_DEFAULT_DEVICE"]

  ENV["DEVICE"] = normalized
  # Ensure forced DEVICE is respected. Focused tests can still override DEVICE
  # inside subprocesses when they need to probe specific backends.
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

def parse_test_devices_arg(raw_devices)
  return [] if raw_devices.nil?

  raw_devices
    .to_s
    .split(",")
    .map(&:strip)
    .reject(&:empty?)
    .map(&:downcase)
end

def run_base_test_task
  with_filtered_gem_platform_warnings do
    Rake::Task[:test_base].reenable
    Rake::Task[:test_base].invoke
  end
end

def run_test_suite_for_device(device = nil)
  if device.nil?
    run_base_test_task
    return
  end

  with_forced_test_device(device) do
    run_base_test_task
  end
end

if ENV.fetch("MLX_STRICT_TESTS", "0") == "1"
  desc "Run tests with strict per-file timeout (set MLX_TEST_TIMEOUT to customize)."
  task :test_base do
    run_strict_test_suite
  end
else
  Rake::TestTask.new(:test_base) do |t|
    t.libs << "test"
    t.pattern = "test/**/*_test.rb"
    t.warning = true
  end
end

desc "Run test suite on cpu+gpu by default. Override devices: rake \"test[cpu]\" or rake \"test[gpu]\"."
task :test, [:devices] do |_task, args|
  devices = parse_test_devices_arg(args[:devices])
  devices = %w[cpu gpu] if devices.empty?

  devices.each do |device|
    puts "==> Running test suite with DEVICE=#{device}"
    run_test_suite_for_device(device)
  end
end

namespace :test do
  desc "Run the full test suite with DEVICE=cpu."
  task :cpu do
    run_test_suite_for_device(:cpu)
  end

  desc "Run the full test suite with DEVICE=gpu."
  task :gpu do
    run_test_suite_for_device(:gpu)
  end

  desc "Build, install, and run tests against the installed gem artifact."
  task :gem do
    Dir.mktmpdir("mlx-ruby-gem-test-") do |tmp_dir|
      gem_home = File.join(tmp_dir, "gems")
      gem_file = File.join(tmp_dir, "mlx.gem")
      run_root = File.join(tmp_dir, "ruby")
      repo_root = File.expand_path("..", __dir__)
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

      install_gemspec_development_dependencies!(File.join(__dir__, "mlx.gemspec"), env: gem_env, chdir: __dir__)
      run_command!(["gem", "build", "mlx.gemspec", "--output", gem_file], env: gem_env, chdir: __dir__)
      run_command!(["gem", "install", "--local", "--no-document", gem_file], env: gem_env, chdir: __dir__)

      spec_info_script = <<~'RUBY'
        spec = Gem::Specification.find_by_name("mlx")
        puts spec.full_gem_path
        puts spec.extension_dir
      RUBY

      spec_info = capture_command_output!(
        [RbConfig.ruby, "-rrubygems", "-e", spec_info_script],
        env: gem_env,
        chdir: __dir__
      ).lines.map(&:strip)

      full_gem_path = spec_info[0]
      extension_dir = spec_info[1]
      if full_gem_path.nil? || full_gem_path.empty? || extension_dir.nil? || extension_dir.empty?
        abort("Failed to resolve installed mlx gem paths from isolated GEM_HOME at #{gem_home}.")
      end

      native_bundle_path = installed_native_bundle_path(extension_dir)
      if native_bundle_path.nil?
        abort("Could not find installed native extension under #{extension_dir}.")
      end

      prepare_installed_gem_test_root(run_root, full_gem_path, native_bundle_path)

      test_env = gem_env.merge(
        "MLX_TEST_RUBY_ROOT" => run_root,
        "MLX_TEST_REPO_ROOT" => repo_root,
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
        chdir: __dir__
      )
      mlx_feature = load_check_output[/^MLX_FEATURE=(.+)$/, 1]
      native_feature = load_check_output[/^NATIVE_FEATURE=(.+)$/, 1]
      if mlx_feature.nil? || native_feature.nil?
        abort("Failed installed-gem load check. Output:\n#{load_check_output}")
      end

      repo_ext_native_prefix = File.join(__dir__, "ext", "mlx", "native")
      if native_feature.start_with?(repo_ext_native_prefix)
        abort("Installed-gem load check failed: native extension resolved to repo ext path #{native_feature}.")
      end

      files = selected_test_files
      failures = []

      files.each do |file|
        print "."
        $stdout.flush
        success = run_test_file_with_timeout(
          file,
          command: [RbConfig.ruby, "-Itest", file],
          chdir: __dir__,
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
      abort "Installed gem test run failed."
    end
  end
end

desc "Build native extension."
task :build do
  ext_dir = File.expand_path("ext/mlx", __dir__)
  make = ENV.fetch("MAKE", RbConfig::CONFIG["MAKE"] || "make")

  sh RbConfig.ruby, "extconf.rb", chdir: ext_dir
  sh make, chdir: ext_dir
end

namespace :docs do
  desc "Build documentation (Doxygen + Sphinx HTML)."
  task :build do
    docs_dir = File.expand_path("docs", __dir__)
    make = ENV.fetch("MAKE", RbConfig::CONFIG["MAKE"] || "make")

    sh "doxygen", chdir: docs_dir
    sh make, "html", chdir: docs_dir
  end
end

namespace :gem do
  desc "Build gem package from mlx.gemspec."
  task :build do
    sh "gem", "build", "mlx.gemspec", chdir: __dir__
  end

  desc "Bump gem version by 0.0.0.1 in lib/mlx/version.rb."
  task :bump do
    version_file = File.expand_path("lib/mlx/version.rb", __dir__)
    content = File.read(version_file)
    version_pattern = /^(\s*VERSION\s*=\s*")([^"]+)(")\s*$/
    match = content.match(version_pattern)

    raise "Could not find VERSION assignment in #{version_file}" unless match

    old_version = match[2]
    segments = old_version.split(".")
    unless segments.all? { |segment| segment.match?(/\A\d+\z/) } && segments.length <= 4
      raise "Expected VERSION in numeric dotted format with up to 4 segments, got #{old_version.inspect}"
    end

    numeric_segments = segments.map(&:to_i)
    numeric_segments << 0 while numeric_segments.length < 4
    numeric_segments[3] += 1
    new_version = numeric_segments.join(".")

    updated = content.sub(version_pattern) { "#{Regexp.last_match(1)}#{new_version}#{Regexp.last_match(3)}" }
    File.write(version_file, updated)

    puts "Bumped version: #{old_version} -> #{new_version}"
  end
end

namespace :benchmark do
  MODELS = %i[transformer cnn mlp rnn karpathy_gpt2].freeze

  def self.requirements_path
    File.join(__dir__, "requirements.txt")
  end

  def self.python_bin
    ENV.fetch("PYTHON", "python3")
  end

  def self.task_class
    require_relative "tasks/benchmark_task"
    BenchmarkTask
  end

  def self.normalize_device(raw_device)
    compute_device = raw_device.to_s.downcase
    compute_device = "gpu" if compute_device == "metal"
    unless %w[cpu gpu].include?(compute_device)
      raise "Invalid DEVICE='#{raw_device}'. Use cpu or gpu."
    end
    compute_device
  end

  def self.base_options
    benchmark_class = task_class
    {
      iterations: ENV.fetch("ITERATIONS", benchmark_class::DEFAULT_ITERATIONS).to_i,
      warmup: ENV.fetch("WARMUP", benchmark_class::DEFAULT_WARMUP).to_i,
      batch_size: ENV.fetch("BATCH", benchmark_class::DEFAULT_BATCH_SIZE).to_i,
      sequence_length: ENV.fetch("SEQUENCE_LENGTH", benchmark_class::DEFAULT_SEQUENCE_LENGTH).to_i,
      target_sequence_length: ENV.fetch("TARGET_SEQUENCE_LENGTH", benchmark_class::DEFAULT_TARGET_SEQUENCE_LENGTH).to_i,
      dims: ENV.fetch("DIMENSIONS", benchmark_class::DEFAULT_DIMS).to_i,
      num_heads: ENV.fetch("HEADS", benchmark_class::DEFAULT_HEADS).to_i,
      num_layers: ENV.fetch("LAYERS", benchmark_class::DEFAULT_LAYERS).to_i,
      python_bin: python_bin
    }
  end

  def self.single_device_options
    raw_device = ENV.fetch("DEVICE", "gpu")
    base_options.merge(compute_device: normalize_device(raw_device))
  end

  def self.matrix_devices
    ENV.fetch("BENCHMARK_DEVICES", "cpu,gpu").split(",").map do |raw|
      normalize_device(raw.strip)
    end.map(&:to_sym).uniq
  end

  desc "Install Python benchmark dependencies into the active Python from requirements.txt."
  task :deps do
    requirements = requirements_path

    raise "Missing requirements file: #{requirements}" unless File.exist?(requirements)
    sh python_bin, "-m", "pip", "install", "-r", requirements
  end

  desc "Compare Ruby and Python transformer implementations."
  task :transformer do
    task = task_class.new(**single_device_options)
    task.run(model: :transformer)
  end

  desc "Compare Ruby and Python CNN implementations."
  task :cnn do
    task = task_class.new(**single_device_options)
    task.run(model: :cnn)
  end

  desc "Compare Ruby and Python MLP implementations."
  task :mlp do
    task = task_class.new(**single_device_options)
    task.run(model: :mlp)
  end

  desc "Compare Ruby and Python RNN implementations."
  task :rnn do
    task = task_class.new(**single_device_options)
    task.run(model: :rnn)
  end

  desc "Compare Ruby and Python GPT-2 implementation (Karpathy tiny-shakespeare full training loop)."
  task :karpathy_gpt2 do
    task = task_class.new(**single_device_options)
    task.run(model: :karpathy_gpt2)
  end

  desc "Run all configured benchmarks on cpu and gpu, then print a final comparison table."
  task :all do
    benchmark_class = task_class
    devices = matrix_devices
    results_by_model = MODELS.each_with_object({}) { |model, out| out[model] = {} }

    devices.each do |device|
      puts "== Running benchmark suite on #{device} =="
      task = benchmark_class.new(**base_options.merge(compute_device: device))
      MODELS.each do |model|
        results_by_model[model][device] = task.run(
          model: model,
          enforce_parity: false,
          print_summary: true
        )
      end
    end

    benchmark_class.print_dual_device_table(results_by_model)

    failed_rows = MODELS.select do |model|
      devices.any? { |device| !results_by_model[model][device]["ok"] }
    end

    unless failed_rows.empty?
      raise "Benchmark matrix had failures for: #{failed_rows.join(', ')}"
    end
  end
end

desc "Alias for benchmark:all."
task benchmark: "benchmark:all"

task default: :test
