# frozen_string_literal: true

require "json"
require "open3"
require "fileutils"
require "rbconfig"
require "tmpdir"
require "stringio"
LIB_ROOT = File.expand_path("../lib", __dir__)
EXAMPLES_ROOT = File.expand_path("../examples/benchmark", __dir__)

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
  DEFAULT_DTYPE = :float32
  WEBGPU_DEFAULT_TIMEOUT_SECONDS = 180
  WEBGPU_DEFAULT_BENCHMARK_WARMUP = 1
  WEBGPU_DEFAULT_BENCHMARK_MEASURE = 3
  WEBGPU_OUTPUT_ABS_TOLERANCE = 1.0
  WEBGPU_OUTPUT_REL_TOLERANCE = 1e-5
  REPO_ROOT = File.expand_path("..", __dir__).freeze
  LOCAL_MODELS = %w[transformer cnn mlp rnn karpathy_gpt2].freeze
  LOCAL_MODEL_SET = LOCAL_MODELS.each_with_object({}) { |name, out| out[name] = true }.freeze
  LOCAL_MODEL_SYMBOLS = LOCAL_MODELS.map(&:to_sym).freeze
  EXAMPLES_SUBMODULE = File.join(REPO_ROOT, "mlx-ruby-examples").freeze

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
    self.class.ensure_benchmark_runtime!
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
    @dtype = self.class.default_dtype
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
    compatibility = MLX::GraphIR.webgpu_compatibility_report(fixture.fetch(:payload))
    unless compatibility.fetch("unsupported_nodes").zero?
      unsupported = compatibility.fetch("unsupported_ops")
      raise "WebGPU export does not support #{model_name}: unsupported ops #{unsupported.inspect}"
    end

    result = nil
    Dir.mktmpdir("mlx-ruby-webgpu-benchmark-") do |dir|
      harness_dir = File.join(dir, "#{model_name}_webgpu")
      MLX::GraphIR.export_onnx_webgpu_harness(
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
        MLX::GraphIR.smoke_test_onnx_webgpu_harness(
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
        dtype: @dtype
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :cnn
      example = BenchmarkExamples::CnnExample.new(
        batch_size: @batch_size,
        dtype: @dtype
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :mlp
      example = BenchmarkExamples::MlpExample.new(
        batch_size: @batch_size,
        dims: @dims,
        dtype: @dtype
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :rnn
      example = BenchmarkExamples::RnnExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        dtype: @dtype
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :karpathy_gpt2
      example = BenchmarkExamples::Gpt2MiniExample.new(
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

    payload = JSON.parse(MLX::GraphIR.export_graph_ir(StringIO.new, trace, seed))
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
      /no available backend found/i,
      /failed to initialize onnx runtime session for providers webgpu/i
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
    status =
      if result.fetch("skipped")
        "SKIPPED"
      elsif result.fetch("ok")
        "OK"
      else
        "FAILED"
      end

    puts format(
      "LO  %-4s %-28s provider=%-8s steady_ms=%-10s diff=%-10s %s",
      compute_device_name.upcase,
      result.fetch("model"),
      (result["provider"] || "-"),
      format_ms_for_summary(result["steady_state_inference_latency_ms"]),
      format_diff_for_summary(result["output_max_abs_diff"]),
      status
    )
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
        dtype: @dtype
      )
    when :cnn
      BenchmarkExamples::CnnExample.new(
        batch_size: @batch_size,
        dtype: @dtype
      )
    when :mlp
      BenchmarkExamples::MlpExample.new(
        batch_size: @batch_size,
        dims: @dims,
        dtype: @dtype
      )
    when :rnn
      BenchmarkExamples::RnnExample.new(
        batch_size: @batch_size,
        sequence_length: @sequence_length,
        dims: @dims,
        dtype: @dtype
      )
    when :karpathy_gpt2
      BenchmarkExamples::Gpt2MiniExample.new(
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

    @warmup.times do
      output = example.run_step
      MLX::Core.eval(output)
    end

    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    @iterations.times do
      output = example.run_step
      MLX::Core.eval(output)
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
    File.join(@repo_root, "test", "fixtures", "karpathy.txt")
  end

  def benchmark_python(model_name)
    ensure_python_mlx_available!

    script_path = python_script_path(model_name)
    command = [@python_bin, script_path, *python_script_args(model_name)]

    output_lines = []
    status = nil
    Open3.popen2e(*command, chdir: @repo_root) do |_stdin, stream, wait_thr|
      while (line = stream.gets)
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
    ruby_status:,
    python_status:,
    ruby_error:,
    python_error:,
    parity_failures:
  )
    rb_seconds = ruby_result && ruby_result["average_ms"] ? ruby_result.fetch("average_ms") / 1000.0 : nil
    py_seconds = python_result && python_result["average_ms"] ? python_result.fetch("average_ms") / 1000.0 : nil
    rb_per_py = if rb_seconds && py_seconds && !py_seconds.zero?
      rb_seconds / py_seconds
    end

    ok = ruby_status.zero? && python_status.zero? && parity_failures.empty?
    status = ok ? "OK" : "FAILED"
    puts format(
      "LO  %-4s %-28s py=%-10s rb=%-10s rb/py=%-8s %s",
      compute_device_name.upcase,
      model_name.to_s,
      format_seconds_for_summary(py_seconds),
      format_seconds_for_summary(rb_seconds),
      format_ratio_for_summary(rb_per_py),
      status
    )

    return if ok

    puts "  ruby_status: #{ruby_status}"
    puts "  python_status: #{python_status}"
    puts "  ruby_error: #{ruby_error}" if ruby_error
    puts "  python_error: #{python_error}" if python_error
    puts "  parity mismatches: #{parity_failures.join(', ')}" unless parity_failures.empty?
    puts
  end

  def format_seconds_for_summary(value)
    return "n/a" if value.nil?

    format("%.6f", value)
  end

  def format_ratio_for_summary(value)
    return "n/a" if value.nil?

    format("%.2fx", value)
  end

  def format_ms_for_summary(value)
    return "n/a" if value.nil?

    format("%.3f", value)
  end

  def format_diff_for_summary(value)
    return "n/a" if value.nil?

    format("%.6f", value)
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

  def self.requirements_path
    File.join(REPO_ROOT, "requirements.txt")
  end

  def self.ensure_benchmark_runtime!
    return if @benchmark_runtime_loaded

    $LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)
    require "mlx"
    Dir[File.join(EXAMPLES_ROOT, "*.rb")].sort.each { |path| require path }
    @benchmark_runtime_loaded = true
  end

  def self.default_dtype
    ensure_benchmark_runtime!
    MLX::Core.public_send(DEFAULT_DTYPE)
  end

  def self.python_bin
    ENV.fetch("PYTHON", "python3")
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
    {
      iterations: benchmark_runs,
      warmup: benchmark_warmup,
      batch_size: ENV.fetch("BATCH", DEFAULT_BATCH_SIZE).to_i,
      sequence_length: ENV.fetch("SEQUENCE_LENGTH", DEFAULT_SEQUENCE_LENGTH).to_i,
      target_sequence_length: ENV.fetch("TARGET_SEQUENCE_LENGTH", DEFAULT_TARGET_SEQUENCE_LENGTH).to_i,
      dims: ENV.fetch("DIMENSIONS", DEFAULT_DIMS).to_i,
      num_heads: ENV.fetch("HEADS", DEFAULT_HEADS).to_i,
      num_layers: ENV.fetch("LAYERS", DEFAULT_LAYERS).to_i,
      python_bin: python_bin
    }
  end

  def self.benchmark_runs
    if ENV.key?("RUNS")
      ENV.fetch("RUNS").to_i
    elsif ENV.key?("ITERATIONS")
      ENV.fetch("ITERATIONS").to_i
    else
      DEFAULT_ITERATIONS
    end
  end

  def self.benchmark_warmup
    if ENV.key?("WARMUP")
      ENV.fetch("WARMUP").to_i
    else
      DEFAULT_WARMUP
    end
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
    {
      timeout_seconds: ENV.fetch("WEBGPU_TIMEOUT", WEBGPU_DEFAULT_TIMEOUT_SECONDS).to_i,
      benchmark_warmup_runs: ENV.fetch("WEBGPU_WARMUP", benchmark_warmup).to_i,
      benchmark_measure_runs: ENV.fetch("WEBGPU_MEASURE", benchmark_runs).to_i,
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
    local_rubylib = File.join(REPO_ROOT, "lib")
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
      [RbConfig.ruby, "-I#{File.join(REPO_ROOT, 'lib')}", "-e", check_script],
      env: env,
      chdir: EXAMPLES_SUBMODULE
    )

    mlx_rb = output[/^MLX_RB=(.+)$/, 1]
    native = output[/^MLX_NATIVE=(.+)$/, 1]
    expected_mlx_rb = File.expand_path(File.join(REPO_ROOT, "lib", "mlx.rb"))
    expected_native_prefix = File.expand_path(File.join(REPO_ROOT, "ext", "mlx", "native"))

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

  def self.selection_from_task_args(args)
    resolve_model_selection(models_argument_from_task_args(args))
  end

  def self.examples_runs
    benchmark_runs
  end

  def self.examples_warmup
    benchmark_warmup
  end

  def self.examples_timeout
    ENV.fetch("BENCH_TIMEOUT", "900").to_i
  end

  def self.run_local_device_benchmarks!(local_models:, device:)
    return if local_models.empty?

    options = base_options.merge(compute_device: device)
    puts "Local adapter: device=#{device} runs=#{options.fetch(:iterations)} warmup=#{options.fetch(:warmup)}"
    benchmark = new(**options)
    failures = []
    local_models.each do |model_name|
      result = benchmark.run(model: model_name, enforce_parity: false, print_summary: true)
      failures << model_name unless result.fetch("ok")
    end
    raise "Local #{device} benchmark failures: #{failures.join(', ')}" unless failures.empty?
  end

  def self.run_local_webgpu_benchmarks!(local_models:)
    return if local_models.empty?

    options = webgpu_options
    puts "Local adapter: device=#{webgpu_compute_device} runs=#{options.fetch(:benchmark_measure_runs)} warmup=#{options.fetch(:benchmark_warmup_runs)}"
    benchmark = new(**base_options.merge(compute_device: webgpu_compute_device))
    failures = []
    local_models.each do |model_name|
      result = benchmark.run_webgpu(model: model_name, **options)
      failures << model_name unless result.fetch("ok")
    end
    raise "Local WebGPU benchmark failures: #{failures.join(', ')}" unless failures.empty?
  end

  def self.run_examples_benchmarks!(example_models:, devices:)
    return if example_models.empty?

    ensure_examples_submodule!
    env = examples_env
    verify_local_mlx_resolution!(env: env)
    require_relative "../examples/benchmark/benchmark_mlx_examples"

    devices.each do |device|
      adapter = ExamplesModelsBenchmarkAdapter.new(
        repo_root: REPO_ROOT,
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
    require_relative "../examples/benchmark/benchmark_mlx_examples"

    adapter = ExamplesModelsBenchmarkAdapter.new(
      repo_root: REPO_ROOT,
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

  def self.run_cpu_task(args)
    selection = selection_from_task_args(args)
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :cpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[cpu])
  end

  def self.run_gpu_task(args)
    selection = selection_from_task_args(args)
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :gpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[gpu])
  end

  def self.run_webgpu_task(args)
    selection = selection_from_task_args(args)
    run_local_webgpu_benchmarks!(local_models: selection.fetch(:local))
    run_examples_webgpu_benchmarks!(example_models: selection.fetch(:examples))
  end

  def self.run_all_task(args)
    selection = selection_from_task_args(args)
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :cpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[cpu])
    run_local_device_benchmarks!(local_models: selection.fetch(:local), device: :gpu)
    run_examples_benchmarks!(example_models: selection.fetch(:examples), devices: %w[gpu])
    run_local_webgpu_benchmarks!(local_models: selection.fetch(:local))
    run_examples_webgpu_benchmarks!(example_models: selection.fetch(:examples))
  end

  def self.run_top_level_task(args)
    models_argument = models_argument_from_task_args(args)
    Rake::Task["benchmark:all"].reenable
    Rake::Task["benchmark:all"].invoke(models_argument)
  end

  def self.install_dependencies!
    requirements = requirements_path
    raise "Missing requirements file: #{requirements}" unless File.exist?(requirements)

    run_command!([python_bin, "-m", "pip", "install", "-r", requirements], chdir: REPO_ROOT)
  end

  def self.run_graph_ir_coverage!
    script = File.join(REPO_ROOT, "test", "parity", "scripts", "generate_graph_ir_webgpu_coverage_report.rb")
    run_command!([RbConfig.ruby, script], chdir: REPO_ROOT)
  end

  def self.unbundled_env(overrides = {})
    env = {}
    ENV.each_key do |key|
      env[key] = nil if key.start_with?("BUNDLE_") || key.start_with?("BUNDLER_")
    end

    rubyopt_tokens = ENV.fetch("RUBYOPT", "").split
    rubyopt_tokens.reject! { |token| token.include?("bundler/setup") }
    env["RUBYOPT"] = rubyopt_tokens.empty? ? nil : rubyopt_tokens.join(" ")
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
end
