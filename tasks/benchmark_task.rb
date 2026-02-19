# frozen_string_literal: true

require "json"
require "open3"
require "fileutils"
require "tmpdir"
require "stringio"
LIB_ROOT = File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)
require "mlx"
EXAMPLES_ROOT = File.expand_path("../examples/benchmark", __dir__)
Dir[File.join(EXAMPLES_ROOT, "*.rb")].sort.each { |path| require path }

class BenchmarkTask
  CNN_CHANNELS = 3
  CNN_HEIGHT = 64
  CNN_WIDTH = 64
  CNN_CLASSES = 1024
  MLP_FACTOR = 4
  RNN_HIDDEN_MULTIPLIER = 2

  DEFAULT_ITERATIONS = 50
  DEFAULT_WARMUP = 10
  DEFAULT_BATCH_SIZE = 8
  DEFAULT_SEQUENCE_LENGTH = 128
  DEFAULT_TARGET_SEQUENCE_LENGTH = 64
  DEFAULT_DIMS = 256
  DEFAULT_HEADS = 8
  DEFAULT_LAYERS = 4
  DEFAULT_DTYPE = MLX::Core.float32
  WEBGPU_DEFAULT_TIMEOUT_SECONDS = 180
  WEBGPU_DEFAULT_BENCHMARK_WARMUP = 1
  WEBGPU_DEFAULT_BENCHMARK_MEASURE = 3
  WEBGPU_OUTPUT_ABS_TOLERANCE = 1.0
  WEBGPU_OUTPUT_REL_TOLERANCE = 1e-5

  def initialize(
    iterations: DEFAULT_ITERATIONS,
    warmup: DEFAULT_WARMUP,
    batch_size: DEFAULT_BATCH_SIZE,
    sequence_length: DEFAULT_SEQUENCE_LENGTH,
    target_sequence_length: DEFAULT_TARGET_SEQUENCE_LENGTH,
    dims: DEFAULT_DIMS,
    num_heads: DEFAULT_HEADS,
    num_layers: DEFAULT_LAYERS,
    compute_device: :gpu,
    python_bin: "python3"
  )
    @iterations = iterations
    @warmup = warmup
    @batch_size = batch_size
    @sequence_length = sequence_length
    @target_sequence_length = target_sequence_length
    @dims = dims
    @num_heads = num_heads
    @num_layers = num_layers
    @compute_device = parse_compute_device(compute_device)
    @python_bin = python_bin
    @repo_root = File.expand_path("..", __dir__)
  end

  def run(model: :transformer, enforce_parity: true, print_summary: true)
    model_name = model.to_sym
    raise "Unknown benchmark model: #{model_name}" unless available_models.include?(model_name)

    ruby_result, ruby_status, ruby_error = run_with_status do
      benchmark_ruby(model_name)
    end
    python_result, python_status, python_error = run_with_status do
      benchmark_python(model_name)
    end

    parity_checks = parity_checks_for(ruby_result, python_result)
    parity_failures = parity_checks.select { |_key, value| value != true }.keys
    ok = ruby_status.zero? && python_status.zero? && parity_failures.empty?
    speedup = if ruby_result && python_result
      python_result.fetch("average_ms") / ruby_result.fetch("average_ms")
    end

    print_run_summary(
      model_name,
      ruby_result: ruby_result,
      python_result: python_result,
      speedup: speedup,
      ruby_status: ruby_status,
      python_status: python_status,
      ruby_error: ruby_error,
      python_error: python_error,
      parity_failures: parity_failures
    ) if print_summary

    result = {
      "model" => model_name.to_s,
      "device" => compute_device_name,
      "ruby" => ruby_result,
      "python" => python_result,
      "python_per_ruby" => speedup,
      "ruby_status" => ruby_status,
      "python_status" => python_status,
      "ruby_error" => ruby_error,
      "python_error" => python_error,
      "checks" => parity_checks,
      "ok" => ok
    }

    return result unless enforce_parity
    return result if ok

    raise parity_failure_message(model_name, ruby_status, python_status, ruby_error, python_error, parity_failures)
  end

  def run_webgpu(
    model: :transformer,
    timeout_seconds: WEBGPU_DEFAULT_TIMEOUT_SECONDS,
    benchmark_warmup_runs: WEBGPU_DEFAULT_BENCHMARK_WARMUP,
    benchmark_measure_runs: WEBGPU_DEFAULT_BENCHMARK_MEASURE,
    require_webgpu: false,
    print_summary: true
  )
    model_name = model.to_sym
    raise "Unknown benchmark model: #{model_name}" unless available_models.include?(model_name)

    fixture = build_webgpu_fixture(model_name)
    compatibility = MLX::Core.graph_ir_webgpu_compatibility_report(fixture.fetch(:payload))
    unless compatibility.fetch("unsupported_nodes").zero?
      unsupported = compatibility.fetch("unsupported_ops")
      raise "WebGPU export does not support #{model_name}: unsupported ops #{unsupported.inspect}"
    end

    result = nil
    Dir.mktmpdir("mlx-ruby-webgpu-benchmark-") do |dir|
      harness_dir = File.join(dir, "#{model_name}_webgpu")
      MLX::Core.export_onnx_webgpu_harness(
        harness_dir,
        fixture.fetch(:payload),
        model_name: "benchmark_#{model_name}",
        execution_providers: ["webgpu"],
        benchmark_warmup_runs: benchmark_warmup_runs,
        benchmark_measure_runs: benchmark_measure_runs
      )
      File.binwrite(
        File.join(harness_dir, "inputs.example.json"),
        JSON.pretty_generate(fixture.fetch(:feeds))
      )

      telemetry = begin
        MLX::Core.smoke_test_onnx_webgpu_harness(
          harness_dir,
          timeout_seconds: timeout_seconds,
          mock_ort: false,
          local_ort: true
        )
      rescue RuntimeError => e
        if webgpu_unavailable_error?(e.message) && !require_webgpu
          result = {
            "model" => model_name.to_s,
            "provider" => "webgpu",
            "skipped" => true,
            "skip_reason" => e.message.lines.first&.strip || e.message,
            "ok" => true
          }
          print_webgpu_run_summary(result) if print_summary
          return result
        end
        raise
      end

      outputs = telemetry.fetch("sample_outputs")
      actual = outputs.values.first
      expected = fixture.fetch(:expected)
      max_diff = max_abs_diff(expected, actual)
      max_ref = max_abs_value(expected)
      tolerance = [WEBGPU_OUTPUT_ABS_TOLERANCE, max_ref * WEBGPU_OUTPUT_REL_TOLERANCE].max
      output_parity_ok = max_diff <= tolerance

      result = {
        "model" => model_name.to_s,
        "provider" => telemetry.fetch("selected_provider"),
        "requested_providers" => telemetry.fetch("requested_providers"),
        "fallback_used" => telemetry.fetch("fallback_used"),
        "model_load_latency_ms" => telemetry.fetch("model_load_latency_ms"),
        "first_inference_latency_ms" => telemetry.fetch("first_inference_latency_ms"),
        "steady_state_inference_latency_ms" => telemetry.fetch("steady_state_inference_latency_ms"),
        "run_timings_ms" => telemetry.fetch("run_timings_ms"),
        "output_parity_ok" => output_parity_ok,
        "output_max_abs_diff" => max_diff,
        "output_tolerance" => tolerance,
        "skipped" => false,
        "ok" => output_parity_ok && telemetry.fetch("selected_provider") == "webgpu" && telemetry.fetch("fallback_used") == false
      }
      print_webgpu_run_summary(result) if print_summary
    end

    raise "WebGPU benchmark parity failed for #{model_name}" if require_webgpu && result && result.fetch("ok") != true

    result
  end

  private

  PARITY_KEYS = %w[input_shape output_shape input_digest reference_output_digest path_signature].freeze

  def available_models
    [:transformer, :cnn, :mlp, :rnn, :karpathy_gpt2]
  end

  def build_webgpu_fixture(model_name)
    seed = MLX::Core.array([0.0], MLX::Core.float32)
    trace = nil
    expected = nil

    case model_name
    when :transformer
      example = BenchmarkExamples::TransformerExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        target_sequence_length: @target_sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        dtype: DEFAULT_DTYPE
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :cnn
      example = BenchmarkExamples::CnnExample.new(
        batch_size: @batch_size,
        dtype: DEFAULT_DTYPE
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :mlp
      example = BenchmarkExamples::MlpExample.new(
        batch_size: @batch_size,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :rnn
      example = BenchmarkExamples::RnnExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :karpathy_gpt2
      example = BenchmarkExamples::KarpathyGpt2Example.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        repo_root: @repo_root
      )
      model = example.instance_variable_get(:@model)
      sample_input, = example.send(:batch_for_step, 0)
      trace = ->(_seed) { model.call(sample_input) }
      expected = model.call(sample_input).to_a
    else
      raise "Unknown benchmark model: #{model_name}"
    end

    payload = JSON.parse(MLX::Core.export_graph_ir(StringIO.new, trace, seed))
    input_name = payload.fetch("inputs").first.fetch("name")

    {
      payload: payload,
      feeds: { input_name => seed.to_a },
      expected: expected
    }
  end

  def webgpu_unavailable_error?(message)
    text = message.to_s
    [
      /navigator\.gpu/i,
      /webgpu[^\n]*not[^\n]*(available|supported|enabled)/i,
      /gpu adapter/i,
      /failed to create gpu/i,
      /no available backend found/i
    ].any? { |pattern| pattern.match?(text) }
  end

  def max_abs_diff(expected, actual)
    if expected.is_a?(Array)
      return 0.0 if expected.empty? && actual.empty?

      expected.zip(actual).map { |exp_item, act_item| max_abs_diff(exp_item, act_item) }.max
    else
      (expected.to_f - actual.to_f).abs
    end
  end

  def max_abs_value(value)
    if value.is_a?(Array)
      return 0.0 if value.empty?

      value.map { |item| max_abs_value(item) }.max
    else
      value.to_f.abs
    end
  end

  def print_webgpu_run_summary(result)
    puts "Benchmark (webgpu): #{result.fetch('model')}"
    if result.fetch("skipped")
      puts "  skipped: #{result.fetch('skip_reason')}"
      puts
      return
    end

    puts "  provider: #{result.fetch('provider')}"
    puts "  fallback_used: #{result.fetch('fallback_used')}"
    puts "  model_load_ms: #{format('%.3f', result.fetch('model_load_latency_ms'))}"
    puts "  first_inference_ms: #{format('%.3f', result.fetch('first_inference_latency_ms'))}"
    puts "  steady_state_ms: #{format('%.3f', result.fetch('steady_state_inference_latency_ms'))}"
    puts "  output_max_abs_diff: #{result.fetch('output_max_abs_diff')}"
    puts "  output_tolerance: #{result.fetch('output_tolerance')}"
    puts "  output_parity_ok: #{result.fetch('output_parity_ok')}"
    puts
  end

  def configuration_summary(model_name)
    case model_name
    when :transformer
      "batch=#{@batch_size}, src_seq=#{@sequence_length}, tgt_seq=#{@target_sequence_length}, " \
        "dims=#{@dims}, heads=#{@num_heads}, layers=#{@num_layers}"
    when :cnn
      "batch=#{@batch_size}, channels=#{CNN_CHANNELS}, height=#{CNN_HEIGHT}, width=#{CNN_WIDTH}, classes=#{CNN_CLASSES}"
    when :mlp
      "batch=#{@batch_size}, input=#{mlp_input_size}, hidden=#{mlp_hidden_size}, output=#{mlp_output_size}"
    when :rnn
      "batch=#{@batch_size}, seq_len=#{@sequence_length}, input=#{@dims}, hidden=#{rnn_hidden_size}"
    when :karpathy_gpt2
      "batch=#{@batch_size}, block=#{@sequence_length}, dims=#{@dims}, heads=#{@num_heads}, layers=#{@num_layers}, vocab=<dataset>"
    else
      ""
    end
  end

  def compute_device_name
    @compute_device.to_s
  end

  def parse_compute_device(value)
    device = value.to_s.downcase.to_sym
    return :cpu if device == :cpu
    return :gpu if device == :metal || device == :gpu

    raise ArgumentError, "Unsupported compute device: #{value}. Use :cpu or :gpu."
  end

  def mlp_input_size
    @dims * MLP_FACTOR
  end

  def mlp_hidden_size
    @dims * MLP_FACTOR
  end

  def mlp_output_size
    @dims
  end

  def rnn_hidden_size
    @dims * RNN_HIDDEN_MULTIPLIER
  end

  def benchmark_ruby(model_name)
    with_compute_device do
      example = build_ruby_example(model_name)
      benchmark_ruby_loop(example)
    end
  end

  def build_ruby_example(model_name)
    case model_name
    when :transformer
      BenchmarkExamples::TransformerExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        target_sequence_length: @target_sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        dtype: DEFAULT_DTYPE
      )
    when :cnn
      BenchmarkExamples::CnnExample.new(
        batch_size: @batch_size,
        dtype: DEFAULT_DTYPE
      )
    when :mlp
      BenchmarkExamples::MlpExample.new(
        batch_size: @batch_size,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
    when :rnn
      BenchmarkExamples::RnnExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        dtype: DEFAULT_DTYPE
      )
    when :karpathy_gpt2
      BenchmarkExamples::KarpathyGpt2Example.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        num_heads: @num_heads,
        num_layers: @num_layers,
        repo_root: @repo_root
      )
    else
      raise "Unknown benchmark model: #{model_name}"
    end
  end

  def benchmark_ruby_loop(example)
    start = nil
    finish = nil
    output = nil
    label = example.label
    warmup_every = log_interval(@warmup)
    iter_every = log_interval(@iterations)

    @warmup.times do |idx|
      output = example.run_step
      MLX::Core.eval(output)
      if (idx + 1) == @warmup || ((idx + 1) % warmup_every).zero?
        puts "[ruby/#{label}] warmup #{idx + 1}/#{@warmup}"
      end
    end

    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @iterations.times do |idx|
      output = example.run_step
      MLX::Core.eval(output)
      if (idx + 1) == @iterations || ((idx + 1) % iter_every).zero?
        puts "[ruby/#{label}] iter #{idx + 1}/#{@iterations}"
      end
    end
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    result = {
      "average_ms" => (finish - start) * 1000.0 / @iterations,
      "iterations" => @iterations,
      "warmup" => @warmup,
      "output_shape" => example.respond_to?(:output_shape) && example.output_shape ? example.output_shape : output.shape
    }

    if example.respond_to?(:verification_input_digest)
      result["input_digest"] = example.verification_input_digest
    end
    if example.respond_to?(:verification_input_shape)
      result["input_shape"] = example.verification_input_shape
    end
    if example.respond_to?(:verification_reference_output_digest)
      result["reference_output_digest"] = example.verification_reference_output_digest
    end
    if example.respond_to?(:benchmark_path_signature)
      result["path_signature"] = example.benchmark_path_signature
    end

    result
  end

  def karpathy_gpt2_dataset_path
    File.join(@repo_root, "benchmark", "fixtures", "karpathy.txt")
  end

  def benchmark_python(model_name)
    ensure_python_mlx_available!

    script_path = python_script_path(model_name)
    command = [@python_bin, script_path, *python_script_args(model_name)]

    output_lines = []
    status = nil
    Open3.popen2e(*command, chdir: @repo_root) do |_stdin, stream, wait_thr|
      while (line = stream.gets)
        puts line
        output_lines << line
      end
      status = wait_thr.value
    end

    unless status&.success?
      raise "Python benchmark failed with exit code #{status&.exitstatus}: #{output_lines.join}"
    end

    result_line = output_lines.reverse.find { |line| !line.strip.empty? }
    raise "Python benchmark did not return JSON output: #{output_lines.join}" unless result_line

    JSON.parse(result_line)
  end

  def run_with_status
    [yield, 0, nil]
  rescue => e
    [nil, status_from_exception(e), e.message]
  end

  def status_from_exception(exception)
    match = exception.message.match(/exit code (\d+)/)
    return match[1].to_i if match

    1
  end

  def parity_checks_for(ruby_result, python_result)
    checks = {}
    PARITY_KEYS.each do |key|
      checks[key] = if ruby_result && python_result
        ruby_result[key] == python_result[key]
      else
        nil
      end
    end
    checks
  end

  def print_run_summary(
    model_name,
    ruby_result:,
    python_result:,
    speedup:,
    ruby_status:,
    python_status:,
    ruby_error:,
    python_error:,
    parity_failures:
  )
    puts "Benchmark (ruby vs python): #{model_name}"
    puts "  configuration: #{configuration_summary(model_name)}"
    puts "  compute device: #{compute_device_name}"
    puts "  iterations: #{@iterations}, warmup: #{@warmup}"

    if ruby_result && python_result
      puts "  ruby_avg_ms:   #{format('%.3f', ruby_result.fetch('average_ms'))}"
      puts "  python_avg_ms: #{format('%.3f', python_result.fetch('average_ms'))}"
      puts "  python/ruby:   #{format('%.2f', speedup)}x"
      puts "  output shape:  #{ruby_result.fetch('output_shape').join('x')} (ruby), " \
        "#{python_result.fetch('output_shape').join('x')} (python)"
    else
      puts "  ruby_status:   #{ruby_status}"
      puts "  python_status: #{python_status}"
      puts "  ruby_error:    #{ruby_error}" if ruby_error
      puts "  python_error:  #{python_error}" if python_error
    end

    unless parity_failures.empty?
      puts "  parity mismatches: #{parity_failures.join(', ')}"
    end
    puts
  end

  def parity_failure_message(model_name, ruby_status, python_status, ruby_error, python_error, parity_failures)
    details = +"Benchmark parity failure for #{model_name} on #{compute_device_name}.\n"
    details << "ruby_status: #{ruby_status}\n"
    details << "python_status: #{python_status}\n"
    details << "ruby_error: #{ruby_error}\n" if ruby_error
    details << "python_error: #{python_error}\n" if python_error
    details << "parity_mismatches: #{parity_failures.join(', ')}\n" unless parity_failures.empty?
    details
  end

  def ensure_python_mlx_available!
    return if @python_mlx_checked

    stdout, stderr, status = Open3.capture3(@python_bin, "-c", "import mlx.core")
    if status.success?
      @python_mlx_checked = true
      return
    end

    error_output = [stdout, stderr].join.strip
    raise <<~MSG
      Python executable '#{@python_bin}' cannot import mlx.core.
      Run `bundle exec rake benchmark:deps` to install requirements into this Python,
      or set PYTHON=/path/to/python for a preconfigured environment.
      Python output:
      #{error_output}
    MSG
  rescue Errno::ENOENT
    raise <<~MSG
      Python executable not found: #{@python_bin}
      Ensure your asdf Python is active (or set PYTHON=/path/to/python),
      then run `bundle exec rake benchmark:deps`.
    MSG
  end

  def python_script_path(model_name)
    file_name = case model_name
    when :transformer
      "transformer_example.py"
    when :cnn
      "cnn_example.py"
    when :mlp
      "mlp_example.py"
    when :rnn
      "rnn_example.py"
    when :karpathy_gpt2
      "karpathy_gpt2_example.py"
    end

    raise "Unknown benchmark model: #{model_name}" if file_name.nil?

    File.join(@repo_root, "examples", "benchmark", "python", file_name)
  end

  def python_script_args(model_name)
    common = [
      "--device", @compute_device.to_s,
      "--batch-size", @batch_size.to_s,
      "--warmup", @warmup.to_s,
      "--iterations", @iterations.to_s
    ]

    case model_name
    when :transformer
      common + [
        "--source-sequence-length", @sequence_length.to_s,
        "--target-sequence-length", @target_sequence_length.to_s,
        "--dims", @dims.to_s,
        "--num-heads", @num_heads.to_s,
        "--num-layers", @num_layers.to_s
      ]
    when :cnn
      common
    when :mlp
      common + ["--dims", @dims.to_s]
    when :rnn
      common + [
        "--sequence-length", @sequence_length.to_s,
        "--dims", @dims.to_s
      ]
    when :karpathy_gpt2
      common + [
        "--sequence-length", @sequence_length.to_s,
        "--dims", @dims.to_s,
        "--num-heads", @num_heads.to_s,
        "--num-layers", @num_layers.to_s,
        "--dataset-path", karpathy_gpt2_dataset_path
      ]
    else
      raise "Unknown benchmark model: #{model_name}"
    end
  end

  def with_compute_device
    default_device = MLX::Core.default_device
    target_device = if @compute_device == :cpu
      MLX::Core.cpu
    else
      MLX::Core.gpu
    end
    MLX::Core.set_default_device(target_device)
    yield
  ensure
    MLX::Core.set_default_device(default_device) if default_device
  end

  def log_interval(total)
    [1, total / 5].max
  end

  public

  def self.print_dual_device_table(results_by_model)
    puts
    puts "| model | py_cpu_s | py_gpu_s | py_cpu/gpu | rb_cpu_s | rb_gpu_s | rb_cpu/gpu | rb/py_cpu | rb/py_gpu | in_shape (cpu/gpu) | in_content (cpu/gpu) | out_shape (cpu/gpu) | out_content (cpu/gpu) |"
    puts "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |"

    results_by_model.each do |model_name, by_device|
      cpu = by_device[:cpu]
      gpu = by_device[:gpu]

      py_cpu_s = seconds_for(cpu, "python")
      py_gpu_s = seconds_for(gpu, "python")
      rb_cpu_s = seconds_for(cpu, "ruby")
      rb_gpu_s = seconds_for(gpu, "ruby")

      row = [
        model_name.to_s,
        format_seconds(py_cpu_s),
        format_seconds(py_gpu_s),
        format_ratio(py_cpu_s, py_gpu_s),
        format_seconds(rb_cpu_s),
        format_seconds(rb_gpu_s),
        format_ratio(rb_cpu_s, rb_gpu_s),
        format_ratio(rb_cpu_s, py_cpu_s),
        format_ratio(rb_gpu_s, py_gpu_s),
        check_pair(cpu, gpu, "input_shape"),
        check_pair(cpu, gpu, "input_digest"),
        check_pair(cpu, gpu, "output_shape"),
        check_pair(cpu, gpu, "reference_output_digest")
      ]

      puts "| #{row.join(' | ')} |"
    end
    puts
  end

  def self.seconds_for(result, side)
    return nil unless result && result[side] && result[side]["average_ms"]

    result[side]["average_ms"] / 1000.0
  end

  def self.format_seconds(value)
    return "n/a" if value.nil?

    format("%.3f", value)
  end

  def self.format_ratio(numerator, denominator)
    return "n/a" if numerator.nil? || denominator.nil? || denominator.zero?

    format("%.2fx", numerator / denominator)
  end

  def self.check_pair(cpu_result, gpu_result, key)
    "#{check_mark(check_value(cpu_result, key))}/#{check_mark(check_value(gpu_result, key))}"
  end

  def self.check_value(result, key)
    return nil unless result && result["checks"].is_a?(Hash)

    result["checks"][key]
  end

  def self.check_mark(value)
    return "-" if value.nil?

    value ? "✓" : "✗"
  end
end
