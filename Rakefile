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
  LOCAL_MODELS = %w[transformer cnn mlp rnn karpathy_gpt2].freeze
  LOCAL_MODEL_SET = LOCAL_MODELS.each_with_object({}) { |name, out| out[name] = true }.freeze
  LOCAL_MODEL_SYMBOLS = LOCAL_MODELS.map(&:to_sym).freeze
  EXAMPLES_SUBMODULE = File.join(__dir__, "mlx-ruby-examples").freeze

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

  def self.webgpu_options
    benchmark_class = task_class
    {
      timeout_seconds: ENV.fetch("WEBGPU_TIMEOUT", benchmark_class::WEBGPU_DEFAULT_TIMEOUT_SECONDS).to_i,
      benchmark_warmup_runs: ENV.fetch("WEBGPU_WARMUP", benchmark_class::WEBGPU_DEFAULT_BENCHMARK_WARMUP).to_i,
      benchmark_measure_runs: ENV.fetch("WEBGPU_MEASURE", benchmark_class::WEBGPU_DEFAULT_BENCHMARK_MEASURE).to_i,
      require_webgpu: ENV["REQUIRE_WEBGPU"] == "1"
    }
  end

  def self.webgpu_compute_device
    raw_device = ENV.fetch("WEBGPU_DEVICE", ENV.fetch("DEVICE", "cpu"))
    normalize_device(raw_device).to_sym
  end

  def self.examples_runner_path
    File.join(EXAMPLES_SUBMODULE, "benchmark", "runner.rb")
  end

  def self.examples_mode
    mode = ENV.fetch("EXAMPLES_MODE", "dsl").to_s.strip
    raise "Invalid EXAMPLES_MODE='#{mode}'. Use dsl or no_dsl." unless %w[dsl no_dsl].include?(mode)

    mode
  end

  def self.examples_env
    local_rubylib = File.join(__dir__, "lib")
    existing_rubylib = ENV["RUBYLIB"].to_s
    rubylib_paths = [local_rubylib]
    rubylib_paths << existing_rubylib unless existing_rubylib.empty?
    unbundled_env("RUBYLIB" => rubylib_paths.join(File::PATH_SEPARATOR))
  end

  def self.ensure_examples_submodule!
    runner = examples_runner_path
    return if File.exist?(runner)

    raise <<~MSG
      Missing examples benchmark runner at #{runner}.
      Run: git submodule update --init --recursive mlx-ruby-examples
    MSG
  end

  def self.verify_local_mlx_resolution!(env:)
    check_script = <<~'RUBY'
      require "mlx"
      mlx_rb = $LOADED_FEATURES.find { |path| path.end_with?("/lib/mlx.rb") || path.end_with?("\\lib\\mlx.rb") }
      native = $LOADED_FEATURES.find do |path|
        path.match?(%r{(?:/|\\)ext(?:/|\\)mlx(?:/|\\)native\.(?:bundle|so)\z}) ||
          path.match?(%r{(?:/|\\)mlx(?:/|\\)native\.(?:bundle|so)\z})
      end
      puts "MLX_RB=#{mlx_rb}"
      puts "MLX_NATIVE=#{native}"
    RUBY

    output = capture_command_output!(
      [RbConfig.ruby, "-I#{File.join(__dir__, 'lib')}", "-e", check_script],
      env: env,
      chdir: EXAMPLES_SUBMODULE
    )

    mlx_rb = output[/^MLX_RB=(.+)$/, 1]
    native = output[/^MLX_NATIVE=(.+)$/, 1]
    expected_mlx_rb = File.expand_path(File.join(__dir__, "lib", "mlx.rb"))
    expected_native_prefix = File.expand_path(File.join(__dir__, "ext", "mlx", "native"))

    if mlx_rb.nil? || File.expand_path(mlx_rb) != expected_mlx_rb
      raise "Expected local mlx.rb at #{expected_mlx_rb}, got #{mlx_rb.inspect}"
    end
    if native.nil? || !File.expand_path(native).start_with?(expected_native_prefix)
      raise "Expected local native extension under #{expected_native_prefix}, got #{native.inspect}"
    end
  end

  def self.parse_examples_model_specs
    return [] unless File.exist?(examples_runner_path)

    specs = []
    current = {}
    File.foreach(examples_runner_path) do |line|
      stripped = line.strip
      if stripped == "{"
        current = {}
        next
      end

      id_match = line.match(/^\s*id:\s*"([^"]+)",\s*$/)
      if id_match
        current["id"] = id_match[1]
        next
      end

      script_match = line.match(/^\s*ruby_script:\s*"([^"]+)",\s*$/)
      if script_match
        current["ruby_script"] = script_match[1]
        next
      end

      if (stripped == "}," || stripped == "}") && current.key?("id") && current.key?("ruby_script")
        specs << current
        current = {}
      end
    end
    specs
  end

  def self.examples_model_specs
    @examples_model_specs ||= parse_examples_model_specs
  end

  def self.examples_model_names
    examples_model_specs.map { |spec| spec.fetch("id") }
  end

  def self.examples_model_set
    @examples_model_set ||= examples_model_names.each_with_object({}) { |name, out| out[name] = true }
  end

  def self.available_models
    return LOCAL_MODELS.dup if examples_model_names.empty?

    LOCAL_MODELS + examples_model_names
  end

  def self.parse_models_argument(raw_models)
    return [] if raw_models.nil?

    raw_models
      .to_s
      .split(",")
      .map(&:strip)
      .reject(&:empty?)
      .uniq
  end

  def self.resolve_model_selection(raw_models)
    requested = parse_models_argument(raw_models)
    selected =
      if requested.empty?
        available_models
      else
        requested.flat_map do |entry|
          case entry
          when "local"
            LOCAL_MODELS
          when "examples"
            examples_model_names
          else
            entry
          end
        end.uniq
      end
    available = available_models
    unknown = selected.reject { |name| available.include?(name) }
    unless unknown.empty?
      raise "Unknown benchmark model(s): #{unknown.join(', ')}. Available: #{available.join(', ')}"
    end

    local = selected.select { |name| LOCAL_MODEL_SET.key?(name) }.map(&:to_sym)
    examples = selected.select { |name| examples_model_set.key?(name) }
    {
      all: selected,
      local: local,
      examples: examples
    }
  end

  def self.models_argument_from_task_args(args)
    values = []
    primary = args[:models]
    values << primary unless primary.nil? || primary.to_s.strip.empty?
    if args.respond_to?(:extras)
      args.extras.each do |entry|
        values << entry unless entry.nil? || entry.to_s.strip.empty?
      end
    end
    return nil if values.empty?

    values.join(",")
  end

  def self.run_local_device_benchmarks!(local_models:, device:)
    return if local_models.empty?

    benchmark = task_class.new(**base_options.merge(compute_device: device))
    failures = []
    local_models.each do |model_name|
      result = benchmark.run(model: model_name, enforce_parity: false, print_summary: true)
      failures << model_name unless result.fetch("ok")
    end
    raise "Local #{device} benchmark failures: #{failures.join(', ')}" unless failures.empty?
  end

  def self.run_local_webgpu_benchmarks!(local_models:)
    return if local_models.empty?

    benchmark = task_class.new(**base_options.merge(compute_device: webgpu_compute_device))
    failures = []
    local_models.each do |model_name|
      result = benchmark.run_webgpu(model: model_name, **webgpu_options)
      failures << model_name unless result.fetch("ok")
    end
    raise "Local WebGPU benchmark failures: #{failures.join(', ')}" unless failures.empty?
  end

  def self.examples_runs
    ENV.fetch("RUNS", "1").to_i
  end

  def self.examples_warmup
    ENV.fetch("WARMUP", "0").to_i
  end

  def self.examples_timeout
    ENV.fetch("BENCH_TIMEOUT", "900").to_i
  end

  def self.run_examples_benchmarks!(example_models:, devices:)
    return if example_models.empty?

    ensure_examples_submodule!
    env = examples_env
    verify_local_mlx_resolution!(env: env)
    require_relative "tasks/examples_models_benchmark_adapter"

    devices.each do |device|
      adapter = ExamplesModelsBenchmarkAdapter.new(
        repo_root: __dir__,
        submodule_root: EXAMPLES_SUBMODULE,
        device: device,
        runs: examples_runs,
        warmup: examples_warmup,
        timeout: examples_timeout,
        mode: examples_mode,
        env: env,
        python_bin: ENV["PYTHON_BIN"] || ENV["PYTHON"],
        ruby_bin: RbConfig.ruby
      )
      adapter.run(models: example_models, print_summary: true)
    end
  end

  def self.run_examples_webgpu_benchmarks!(example_models:)
    return if example_models.empty?

    ensure_examples_submodule!
    env = examples_env
    verify_local_mlx_resolution!(env: env)
    require_relative "tasks/examples_models_benchmark_adapter"

    adapter = ExamplesModelsBenchmarkAdapter.new(
      repo_root: __dir__,
      submodule_root: EXAMPLES_SUBMODULE,
      device: webgpu_compute_device,
      runs: examples_runs,
      warmup: examples_warmup,
      timeout: examples_timeout,
      mode: examples_mode,
      env: env,
      python_bin: ENV["PYTHON_BIN"] || ENV["PYTHON"],
      ruby_bin: RbConfig.ruby
    )
    adapter.run_webgpu(
      models: example_models,
      timeout_seconds: webgpu_options.fetch(:timeout_seconds),
      benchmark_warmup_runs: webgpu_options.fetch(:benchmark_warmup_runs),
      benchmark_measure_runs: webgpu_options.fetch(:benchmark_measure_runs),
      require_webgpu: webgpu_options.fetch(:require_webgpu),
      print_summary: true
    )
  end

  desc "Install Python benchmark dependencies into the active Python from requirements.txt."
  task :deps do
    requirements = requirements_path

    raise "Missing requirements file: #{requirements}" unless File.exist?(requirements)
    sh python_bin, "-m", "pip", "install", "-r", requirements
  end

  desc "Run selected models on cpu. Usage: rake 'benchmark:cpu[local,examples]' or rake 'benchmark:cpu[model_a,model_b]'."
  task :cpu, [:models] do |_task, args|
    selection = resolve_model_selection(models_argument_from_task_args(args))
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :cpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[cpu])
  end

  desc "Run selected models on gpu. Usage: rake 'benchmark:gpu[local,examples]' or rake 'benchmark:gpu[model_a,model_b]'."
  task :gpu, [:models] do |_task, args|
    selection = resolve_model_selection(models_argument_from_task_args(args))
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :gpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[gpu])
  end

  desc "Run selected models on WebGPU/local GPU path. Usage: rake 'benchmark:webgpu[local,examples]' or rake 'benchmark:webgpu[model_a,model_b]'."
  task :webgpu, [:models] do |_task, args|
    selection = resolve_model_selection(models_argument_from_task_args(args))
    run_local_webgpu_benchmarks!(local_models: selection.fetch(:local))
    run_examples_webgpu_benchmarks!(example_models: selection.fetch(:examples))
  end

  desc "Generate GraphIR WebGPU compatibility coverage artifact for benchmark fixtures."
  task :graph_ir_coverage do
    script = File.join(__dir__, "test", "parity", "scripts", "generate_graph_ir_webgpu_coverage_report.rb")
    sh RbConfig.ruby, script
  end

  desc "Run selected models on cpu, gpu, and webgpu. Usage: rake 'benchmark:all[local,examples]' or rake 'benchmark:all[model_a,model_b]'."
  task :all, [:models] do |_task, args|
    selection = resolve_model_selection(models_argument_from_task_args(args))
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :cpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[cpu])
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :gpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[gpu])
    run_local_webgpu_benchmarks!(local_models: selection.fetch(:local))
    run_examples_webgpu_benchmarks!(example_models: selection.fetch(:examples))
  end
end

desc "Run selected benchmarks on cpu, gpu, and webgpu. Usage: rake 'benchmark[local,examples]' or rake 'benchmark[model_a,model_b]'."
task :benchmark, [:models] do |_task, args|
  values = []
  primary = args[:models]
  values << primary unless primary.nil? || primary.to_s.strip.empty?
  if args.respond_to?(:extras)
    args.extras.each do |entry|
      values << entry unless entry.nil? || entry.to_s.strip.empty?
    end
  end
  models_argument = values.empty? ? nil : values.join(",")

  Rake::Task["benchmark:all"].reenable
  Rake::Task["benchmark:all"].invoke(models_argument)
end

task default: :test
