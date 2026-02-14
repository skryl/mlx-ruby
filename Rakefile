# frozen_string_literal: true

require "rake"
require "rake/clean"
require "rake/file_list"
require "rake/testtask"
require "rbconfig"
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

def test_file_list
  Rake::FileList["test/**/*_test.rb"]
end

def run_test_file_with_timeout(file, timeout: strict_test_timeout_seconds)
  command = ["bundle", "exec", "ruby", "-Itest", file]
  pid = Process.spawn(*command, chdir: __dir__)

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

def run_strict_test_suite
  files = test_file_list.to_a
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
  Rake::Task[:test_base].reenable
  Rake::Task[:test_base].invoke
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
  def self.task_class
    require_relative "tasks/benchmark_task"
    BenchmarkTask
  end

  def self.options
    raw_device = ENV.fetch("DEVICE", "gpu").downcase
    compute_device = raw_device == "metal" ? "gpu" : raw_device
    unless %w[cpu gpu].include?(compute_device)
      raise "Invalid DEVICE='#{raw_device}'. Use cpu or gpu."
    end

    {
      iterations: ENV.fetch("ITERATIONS", BenchmarkTask::DEFAULT_ITERATIONS).to_i,
      warmup: ENV.fetch("WARMUP", BenchmarkTask::DEFAULT_WARMUP).to_i,
      batch_size: ENV.fetch("BATCH", BenchmarkTask::DEFAULT_BATCH_SIZE).to_i,
      sequence_length: ENV.fetch("SEQUENCE_LENGTH", BenchmarkTask::DEFAULT_SEQUENCE_LENGTH).to_i,
      target_sequence_length: ENV.fetch("TARGET_SEQUENCE_LENGTH", BenchmarkTask::DEFAULT_TARGET_SEQUENCE_LENGTH).to_i,
      dims: ENV.fetch("DIMENSIONS", BenchmarkTask::DEFAULT_DIMS).to_i,
      num_heads: ENV.fetch("HEADS", BenchmarkTask::DEFAULT_HEADS).to_i,
      num_layers: ENV.fetch("LAYERS", BenchmarkTask::DEFAULT_LAYERS).to_i,
      compute_device: compute_device,
      python_bin: ENV.fetch("PYTHON", "python3")
    }
  end

  desc "Compare Ruby and Python transformer implementations."
  task :transformer do
    task = task_class.new(**options)
    task.run(model: :transformer)
  end

  desc "Compare Ruby and Python CNN implementations."
  task :cnn do
    task = task_class.new(**options)
    task.run(model: :cnn)
  end

  desc "Compare Ruby and Python MLP implementations."
  task :mlp do
    task = task_class.new(**options)
    task.run(model: :mlp)
  end

  desc "Compare Ruby and Python RNN implementations."
  task :rnn do
    task = task_class.new(**options)
    task.run(model: :rnn)
  end

  desc "Compare Ruby and Python GPT-2 implementation (Karpathy tiny-shakespeare full training loop)."
  task :karpathy_gpt2 do
    task = task_class.new(**options)
    task.run(model: :karpathy_gpt2)
  end

  desc "Run all configured benchmarks (transformer, cnn, mlp, rnn, karpathy_gpt2)."
  task all: %i[transformer cnn mlp rnn karpathy_gpt2]
end

task default: :test
