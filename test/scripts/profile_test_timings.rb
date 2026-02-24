# frozen_string_literal: true

require "json"
require "optparse"
require "fileutils"
require "tmpdir"
require "time"

module TestTimingProfiler
  module_function

  REPO_ROOT = File.expand_path("../..", __dir__).freeze
  DEFAULT_THRESHOLD_SECONDS = 30.0
  DEFAULT_TIMINGS_OUT = File.join(REPO_ROOT, "test", "tmp", "reports", "parity", "test_timings.json").freeze
  DEFAULT_SLOW_OUT = File.join(REPO_ROOT, "test", "slow_tests.json").freeze
  TEST_TIMING_LINE = /^\s*([A-Za-z0-9_:]+#\S+)\s*=\s*([0-9]+(?:\.[0-9]+)?) s =/.freeze

  def parse_verbose_log(path)
    timings = {}
    File.foreach(path) do |line|
      match = TEST_TIMING_LINE.match(line)
      next unless match

      test_name = match[1]
      seconds = match[2].to_f
      previous = timings[test_name]
      timings[test_name] = previous.nil? ? seconds : [previous, seconds].max
    end
    sort_hash_by_key(timings)
  end

  def merge_device_timings(cpu_timings, gpu_timings)
    merged = {}
    (cpu_timings.keys | gpu_timings.keys).sort.each do |test_name|
      cpu_seconds = cpu_timings[test_name]
      gpu_seconds = gpu_timings[test_name]
      max_seconds = [cpu_seconds, gpu_seconds].compact.max
      merged[test_name] = {
        "cpu_seconds" => cpu_seconds,
        "gpu_seconds" => gpu_seconds,
        "max_seconds" => max_seconds
      }
    end
    merged
  end

  def build_timing_payload(merged_timings, threshold_seconds:, cpu_log:, gpu_log:)
    {
      "format" => "mlx_test_timings_v1",
      "generated_at" => Time.now.utc.iso8601,
      "threshold_seconds" => threshold_seconds.to_f,
      "devices" => {
        "cpu" => {
          "log_path" => cpu_log,
          "tests_observed" => merged_timings.values.count { |entry| !entry["cpu_seconds"].nil? }
        },
        "gpu" => {
          "log_path" => gpu_log,
          "tests_observed" => merged_timings.values.count { |entry| !entry["gpu_seconds"].nil? }
        }
      },
      "tests" => sort_hash_by_key(merged_timings)
    }
  end

  def build_slow_registry(merged_timings, threshold_seconds:)
    threshold = threshold_seconds.to_f
    slow_tests = {}
    sort_hash_by_key(merged_timings).each do |test_name, entry|
      max_seconds = entry["max_seconds"]
      next if max_seconds.nil?
      next unless max_seconds > threshold

      slow_tests[test_name] = {
        "cpu_seconds" => entry["cpu_seconds"],
        "gpu_seconds" => entry["gpu_seconds"],
        "max_seconds" => max_seconds
      }
    end

    {
      "format" => "mlx_slow_tests_v1",
      "generated_at" => Time.now.utc.iso8601,
      "threshold_seconds" => threshold,
      "tests" => slow_tests
    }
  end

  def run_suite_for_device(device, log_path, repo_root:)
    FileUtils.mkdir_p(File.dirname(log_path))
    File.open(log_path, "w") do |out|
      env = {
        "TESTOPTS" => "--verbose",
        "MLX_TEST_INCLUDE_SLOW" => "1"
      }
      success = system(env, "bundle", "exec", "rake", "test:#{device}", chdir: repo_root, out: out, err: out)
      return if success
    end

    raise "failed to run test suite for device=#{device}; see #{log_path}"
  end

  def write_json(path, payload)
    FileUtils.mkdir_p(File.dirname(path))
    File.write(path, "#{JSON.pretty_generate(payload)}\n")
  end

  def run!(argv = ARGV)
    options = {
      threshold_seconds: DEFAULT_THRESHOLD_SECONDS,
      timings_out: DEFAULT_TIMINGS_OUT,
      slow_out: DEFAULT_SLOW_OUT,
      repo_root: REPO_ROOT,
      run: false
    }

    parser = OptionParser.new do |opts|
      opts.banner = "Usage: ruby test/scripts/profile_test_timings.rb [options]"
      opts.on("--cpu-log PATH", "Path to cpu verbose log") { |value| options[:cpu_log] = value }
      opts.on("--gpu-log PATH", "Path to gpu verbose log") { |value| options[:gpu_log] = value }
      opts.on("--threshold SECONDS", Float, "Slow threshold in seconds (default: #{DEFAULT_THRESHOLD_SECONDS})") do |value|
        options[:threshold_seconds] = value
      end
      opts.on("--timings-out PATH", "Timing artifact output path (default: #{DEFAULT_TIMINGS_OUT})") do |value|
        options[:timings_out] = value
      end
      opts.on("--slow-out PATH", "Slow registry output path (default: #{DEFAULT_SLOW_OUT})") do |value|
        options[:slow_out] = value
      end
      opts.on("--repo-root PATH", "Repository root (default: #{REPO_ROOT})") { |value| options[:repo_root] = value }
      opts.on("--run", "Run cpu/gpu suites to generate logs before parsing") { options[:run] = true }
    end
    parser.parse!(argv)

    cpu_log = options[:cpu_log]
    gpu_log = options[:gpu_log]

    if options[:run] || cpu_log.nil? || gpu_log.nil?
      cpu_log ||= File.join(Dir.tmpdir, "mlx_test_cpu_verbose_profile.log")
      gpu_log ||= File.join(Dir.tmpdir, "mlx_test_gpu_verbose_profile.log")
      run_suite_for_device("cpu", cpu_log, repo_root: options[:repo_root])
      run_suite_for_device("gpu", gpu_log, repo_root: options[:repo_root])
    end

    cpu_timings = parse_verbose_log(cpu_log)
    gpu_timings = parse_verbose_log(gpu_log)
    merged = merge_device_timings(cpu_timings, gpu_timings)
    timing_payload = build_timing_payload(
      merged,
      threshold_seconds: options[:threshold_seconds],
      cpu_log: cpu_log,
      gpu_log: gpu_log
    )
    slow_registry = build_slow_registry(merged, threshold_seconds: options[:threshold_seconds])

    write_json(options[:timings_out], timing_payload)
    write_json(options[:slow_out], slow_registry)

    puts "Wrote timing artifact: #{options[:timings_out]}"
    puts "Wrote slow test registry: #{options[:slow_out]}"
    puts "Slow tests (>#{options[:threshold_seconds]}s): #{slow_registry.fetch('tests').length}"
  end

  def sort_hash_by_key(hash)
    hash.keys.sort.each_with_object({}) do |key, out|
      out[key] = hash[key]
    end
  end
end

if $PROGRAM_NAME == __FILE__
  TestTimingProfiler.run!
end
