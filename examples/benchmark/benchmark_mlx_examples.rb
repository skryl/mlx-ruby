# frozen_string_literal: true

require "digest"
require "json"
require "open3"
require "rbconfig"
require "shellwords"
require "stringio"
require "timeout"
require "tmpdir"

EXAMPLES_ADAPTER_LIB_ROOT = File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift(EXAMPLES_ADAPTER_LIB_ROOT) unless $LOAD_PATH.include?(EXAMPLES_ADAPTER_LIB_ROOT)
require "mlx"

class ExamplesModelsBenchmarkAdapter
  PARITY_KEYS = %w[input_shape_match input_content_match output_shape_match output_content_match].freeze
  EXTRA_PYTHON_PACKAGES = %w[
    mlx
    numpy
    transformers
    huggingface_hub
    tqdm
    sentencepiece
    protobuf
    Pillow
    tiktoken
  ].freeze
  PREFERRED_PYTHON = %w[python3.12 python3.11 python3.10 python3 python].freeze
  WEBGPU_OUTPUT_ABS_TOLERANCE = 1.0
  WEBGPU_OUTPUT_REL_TOLERANCE = 1e-5

  def initialize(
    repo_root:,
    submodule_root:,
    device:,
    runs:,
    warmup:,
    timeout:,
    mode:,
    env: {},
    python_bin: nil,
    ruby_bin: RbConfig.ruby
  )
    @repo_root = repo_root
    @submodule_root = submodule_root
    @device = normalize_device(device)
    @runs = Integer(runs)
    @warmup = Integer(warmup)
    @timeout = Integer(timeout)
    @mode = mode.to_s
    @env = env
    @python_bin = python_bin&.to_s
    @ruby_bin = ruby_bin

    raise ArgumentError, "runs must be >= 1" if @runs < 1
    raise ArgumentError, "warmup must be >= 0" if @warmup < 0
    raise ArgumentError, "timeout must be >= 1" if @timeout < 1
    raise ArgumentError, "mode must be dsl or no_dsl" unless %w[dsl no_dsl].include?(@mode)
  end

  def run(models:, print_summary: true)
    ensure_submodule_present!
    ensure_helpers_present!
    @specs = parse_model_specs
    @specs_by_id = @specs.each_with_object({}) { |spec, out| out[spec.fetch("id")] = spec }

    unknown = models.reject { |model_id| @specs_by_id.key?(model_id) }
    raise ArgumentError, "Unknown examples model(s): #{unknown.join(', ')}" unless unknown.empty?

    @python_command = resolve_python_command
    @python_env_string = Shellwords.join(@python_command)
    puts "Examples adapter: device=#{@device} mode=#{@mode} runs=#{@runs} warmup=#{@warmup}" if print_summary

    failures = []
    results = models.each_with_object({}) do |model_id, out|
      spec = @specs_by_id.fetch(model_id)
      ruby_result = run_ruby_model(spec)
      python_result = run_python_model(model_id)
      parity_failures = parity_failures_for(ruby_result.fetch(:parity))
      ok = ruby_result.fetch(:status).zero? && python_result.fetch(:status).zero? && parity_failures.empty?

      if print_summary
        puts format(
          "EX  %-4s %-28s py=%-10s rb=%-10s rb/py=%-8s %s",
          @device.upcase,
          model_id,
          format_seconds(python_result.fetch(:seconds)),
          format_seconds(ruby_result.fetch(:seconds)),
          format_ratio(ruby_result.fetch(:seconds), python_result.fetch(:seconds)),
          ok ? "OK" : "FAILED"
        )
      end

      out[model_id] = {
        ruby: ruby_result,
        python: python_result,
        parity_failures: parity_failures,
        ok: ok
      }
      failures << model_id unless ok
    end

    return results if failures.empty?

    raise "Examples #{@device} benchmark failures: #{failures.join(', ')}"
  end

  def run_webgpu(
    models:,
    timeout_seconds: 180,
    benchmark_warmup_runs: 1,
    benchmark_measure_runs: 3,
    require_webgpu: false,
    print_summary: true
  )
    ensure_submodule_present!
    ensure_helpers_present!
    @specs = parse_model_specs
    @specs_by_id = @specs.each_with_object({}) { |spec, out| out[spec.fetch("id")] = spec }

    unknown = models.reject { |model_id| @specs_by_id.key?(model_id) }
    raise ArgumentError, "Unknown examples model(s): #{unknown.join(', ')}" unless unknown.empty?

    puts "Examples adapter (webgpu): device=#{@device} mode=#{@mode}" if print_summary
    failures = []
    results = models.each_with_object({}) do |model_id, out|
      spec = @specs_by_id.fetch(model_id)
      capture = capture_onnx_fixture(spec)
      payload = capture.fetch("payload")
      compatibility = MLX::GraphIR.webgpu_compatibility_report(payload)

      if compatibility.fetch("unsupported_nodes").to_i > 0
        unsupported = compatibility.fetch("unsupported_ops")
        unsupported_message = "unsupported ops: #{unsupported.inspect}"
        row = if require_webgpu
          {
            "model" => model_id,
            "unsupported_ops" => unsupported,
            "unsupported_nodes" => compatibility.fetch("unsupported_nodes"),
            "ok" => false,
            "skipped" => false,
            "error" => unsupported_message
          }
        else
          {
            "model" => model_id,
            "unsupported_ops" => unsupported,
            "unsupported_nodes" => compatibility.fetch("unsupported_nodes"),
            "provider" => "webgpu",
            "ok" => true,
            "skipped" => true,
            "skip_reason" => unsupported_message
          }
        end
        out[model_id] = row
        failures << model_id if require_webgpu
        puts "EXW #{model_id.ljust(28)} unsupported #{unsupported.inspect}" if print_summary
        next
      end

      row = run_webgpu_harness_for_capture(
        model_id: model_id,
        capture: capture,
        timeout_seconds: timeout_seconds,
        benchmark_warmup_runs: benchmark_warmup_runs,
        benchmark_measure_runs: benchmark_measure_runs,
        require_webgpu: require_webgpu
      )

      out[model_id] = row
      failures << model_id unless row.fetch("ok")
      if print_summary
        status = row.fetch("skipped") ? "SKIPPED" : (row.fetch("ok") ? "OK" : "FAILED")
        puts format(
          "EXW %-28s provider=%-8s steady_ms=%-10s diff=%-10s %s",
          model_id,
          (row["provider"] || "-"),
          format_ms(row["steady_state_inference_latency_ms"]),
          format_diff(row["output_max_abs_diff"]),
          status
        )
      end
    end

    return results if failures.empty?

    raise "Examples webgpu benchmark failures: #{failures.join(', ')}"
  end

  private

  def normalize_device(raw)
    value = raw.to_s.downcase
    return "cpu" if value == "cpu"
    return "gpu" if %w[gpu metal webgpu].include?(value)

    raise ArgumentError, "unsupported examples benchmark device: #{raw.inspect}"
  end

  def ensure_submodule_present!
    runner = File.join(@submodule_root, "benchmark", "runner.rb")
    return if File.exist?(runner)

    raise "Missing examples benchmark runner at #{runner}"
  end

  def ensure_helpers_present!
    required = [
      File.join(@submodule_root, "benchmark", "python", "run_model.py"),
      File.join(@submodule_root, "benchmark", "ruby", "run_with_dryrun.rb"),
      File.join(@submodule_root, "benchmark", "ruby", "force_cpu.rb"),
      File.join(@submodule_root, "benchmark", "ruby", "force_gpu.rb")
    ]
    missing = required.reject { |path| File.exist?(path) }
    return if missing.empty?

    raise "Examples benchmark helper files missing:\n#{missing.join("\n")}"
  end

  def parse_model_specs
    runner_path = File.join(@submodule_root, "benchmark", "runner.rb")
    specs = []
    current = {}
    File.foreach(runner_path) do |line|
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

    raise "Failed to parse model specs from #{runner_path}" if specs.empty?

    specs
  end

  def resolve_python_command
    explicit = @python_bin.to_s.strip
    unless explicit.empty?
      command = Shellwords.split(explicit)
      return command if executable_python?(command)

      raise "PYTHON_BIN/PYTHON command is not executable: #{explicit.inspect}"
    end

    existing_venv = Dir.glob(File.join(@submodule_root, "tmp", "benchmark", ".venv_*", "bin", "python")).sort.last
    return [existing_venv] if existing_venv && File.executable?(existing_venv)

    ensure_local_python_environment!
  end

  def ensure_local_python_environment!
    base = PREFERRED_PYTHON.map { |candidate| [candidate] }.find { |command| executable_python?(command) }
    raise "Could not locate a Python executable (tried: #{PREFERRED_PYTHON.join(', ')})" if base.nil?

    venv_dir = File.join(@submodule_root, "tmp", "benchmark", ".venv_local_adapter")
    venv_python = File.join(venv_dir, "bin", "python")
    unless File.exist?(venv_python)
      run_command!(base + ["-m", "venv", venv_dir], chdir: @submodule_root, env: @env, timeout: 600, label: "create examples venv")
    end

    stamp_path = File.join(venv_dir, ".adapter_deps.sha256")
    digest = Digest::SHA256.hexdigest(EXTRA_PYTHON_PACKAGES.join(","))
    unless File.exist?(stamp_path) && File.read(stamp_path).strip == digest
      run_command!(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        chdir: @submodule_root,
        env: @env,
        timeout: 1200,
        label: "upgrade pip"
      )
      run_command!(
        [venv_python, "-m", "pip", "install", *EXTRA_PYTHON_PACKAGES],
        chdir: @submodule_root,
        env: @env,
        timeout: 1800,
        label: "install examples benchmark python dependencies"
      )
      File.write(stamp_path, "#{digest}\n")
    end

    [venv_python]
  end

  def executable_python?(command)
    result = run_capture(command + ["-c", "import sys"], chdir: @submodule_root, env: @env, timeout: 30)
    result.fetch(:status).zero?
  end

  def run_python_model(model_id)
    command = @python_command + [
      File.join(@submodule_root, "benchmark", "python", "run_model.py"),
      "--mlx-examples", File.join(@submodule_root, "mlx-examples"),
      "--model", model_id,
      "--device", @device,
      "--warmup", @warmup.to_s,
      "--runs", @runs.to_s
    ]

    result = run_capture(command, chdir: @repo_root, env: @env, timeout: @timeout)
    return failed_result("python", result) unless result.fetch(:status).zero?

    payload = parse_json_payload(result.fetch(:stdout))
    {
      status: 0,
      seconds: Float(payload.fetch("seconds")),
      samples: payload.fetch("samples"),
      payload: payload,
      stderr: result.fetch(:stderr)
    }
  rescue StandardError => e
    {
      status: 1,
      seconds: nil,
      samples: [],
      payload: nil,
      stderr: [result && result[:stderr], "#{e.class}: #{e.message}"].compact.join("\n")
    }
  end

  def run_ruby_model(spec)
    command = [
      @ruby_bin,
      "-I#{File.join(@repo_root, 'lib')}",
      "-r",
      force_device_script,
      File.join(@submodule_root, "benchmark", "ruby", "run_with_dryrun.rb"),
      ruby_script_path(spec)
    ]
    env = @env.merge(
      "PYTHON_BIN" => @python_env_string,
      "MLX_BENCHMARK" => "1",
      "MLX_BENCHMARK_DEVICE" => @device
    )

    samples = []
    parity = nil
    total_iters = @warmup + @runs

    total_iters.times do |idx|
      result = run_capture(command, chdir: @submodule_root, env: env, timeout: @timeout)
      return failed_result("ruby", result) unless result.fetch(:status).zero?

      next if idx < @warmup

      samples << parse_benchmark_seconds(result.fetch(:stdout))
      parsed_parity = parse_ruby_parity(result.fetch(:stdout))
      parity = parsed_parity if parsed_parity
    end

    {
      status: 0,
      seconds: samples.sum / samples.length,
      samples: samples,
      parity: parity,
      stderr: ""
    }
  rescue StandardError => e
    {
      status: 1,
      seconds: nil,
      samples: [],
      parity: nil,
      stderr: e.message
    }
  end

  def capture_onnx_fixture(spec)
    capture_path = File.join(Dir.mktmpdir("examples-onnx-capture-"), "capture.json")
    env = @env.merge(
      "MLX_BENCHMARK" => "1",
      "MLX_BENCHMARK_DEVICE" => @device,
      "MLX_EXAMPLES_SUBMODULE_ROOT" => @submodule_root,
      "MLX_EXAMPLES_ONNX_CAPTURE_FILE" => capture_path
    )
    command = [
      @ruby_bin,
      "-I#{File.join(@repo_root, 'lib')}",
      "-r",
      force_device_script,
      "-r",
      File.join(@repo_root, "examples", "benchmark", "benchmark_mlx_examples", "onnx_capture_hook.rb"),
      File.join(@submodule_root, "benchmark", "ruby", "run_with_dryrun.rb"),
      ruby_script_path(spec)
    ]

    result = run_capture(command, chdir: @submodule_root, env: env, timeout: @timeout)
    unless result.fetch(:status).zero?
      raise "ONNX capture failed for #{spec.fetch('id')}:\nstdout:\n#{result.fetch(:stdout)}\nstderr:\n#{result.fetch(:stderr)}"
    end

    unless File.exist?(capture_path) && !File.zero?(capture_path)
      raise "ONNX capture file was not written for #{spec.fetch('id')}"
    end

    capture = JSON.parse(File.read(capture_path))
    if capture.key?("error")
      raise "ONNX capture hook failed for #{spec.fetch('id')}: #{capture.fetch('error')}"
    end

    payload = capture.fetch("payload")
    node_count = payload.fetch("nodes", []).length
    if node_count <= 0
      raise "ONNX capture produced empty graph for #{spec.fetch('id')}: node_count=#{node_count}"
    end

    capture
  ensure
    File.unlink(capture_path) if capture_path && File.exist?(capture_path)
  end

  def run_webgpu_harness_for_capture(
    model_id:,
    capture:,
    timeout_seconds:,
    benchmark_warmup_runs:,
    benchmark_measure_runs:,
    require_webgpu:
  )
    payload = capture.fetch("payload")
    feeds = capture.fetch("feeds")
    expected_outputs = capture.fetch("expected_outputs")

    Dir.mktmpdir("examples-webgpu-#{model_id.tr('/', '_')}-") do |dir|
      harness_dir = File.join(dir, "harness")
      MLX::GraphIR.export_onnx_webgpu_harness(
        harness_dir,
        payload,
        model_name: "examples_#{model_id.tr('/', '_')}",
        execution_providers: ["webgpu"],
        benchmark_warmup_runs: benchmark_warmup_runs,
        benchmark_measure_runs: benchmark_measure_runs
      )
      File.binwrite(
        File.join(harness_dir, "inputs.example.json"),
        JSON.pretty_generate(feeds)
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
          return {
            "model" => model_id,
            "provider" => "webgpu",
            "skipped" => true,
            "skip_reason" => e.message.lines.first&.strip || e.message,
            "ok" => true
          }
        end
        raise
      end

      sample_outputs = telemetry.fetch("sample_outputs")
      output_names = capture.fetch("output_names")
      resolved_pairs = resolve_output_pairs(output_names, expected_outputs, sample_outputs)
      max_diff = resolved_pairs.map { |pair| max_abs_diff(pair.fetch(:expected), pair.fetch(:actual)) }.max || 0.0
      max_ref = resolved_pairs.map { |pair| max_abs_value(pair.fetch(:expected)) }.max || 0.0
      tolerance = [WEBGPU_OUTPUT_ABS_TOLERANCE, max_ref * WEBGPU_OUTPUT_REL_TOLERANCE].max
      output_ok = max_diff <= tolerance

      {
        "model" => model_id,
        "provider" => telemetry.fetch("selected_provider"),
        "requested_providers" => telemetry.fetch("requested_providers"),
        "fallback_used" => telemetry.fetch("fallback_used"),
        "model_load_latency_ms" => telemetry.fetch("model_load_latency_ms"),
        "first_inference_latency_ms" => telemetry.fetch("first_inference_latency_ms"),
        "steady_state_inference_latency_ms" => telemetry.fetch("steady_state_inference_latency_ms"),
        "run_timings_ms" => telemetry.fetch("run_timings_ms"),
        "output_parity_ok" => output_ok,
        "output_max_abs_diff" => max_diff,
        "output_tolerance" => tolerance,
        "skipped" => false,
        "ok" => output_ok && telemetry.fetch("selected_provider") == "webgpu" && telemetry.fetch("fallback_used") == false
      }
    end
  end

  def resolve_output_pairs(output_names, expected_outputs, sample_outputs)
    sample_keys = sample_outputs.keys
    output_names.each_with_index.map do |name, idx|
      expected = expected_outputs.fetch(name)
      actual_key = if sample_outputs.key?(name)
        name
      elsif idx < sample_keys.length
        sample_keys[idx]
      else
        nil
      end
      raise "missing runtime output for #{name}" if actual_key.nil?

      {
        name: name,
        expected: expected,
        actual: sample_outputs.fetch(actual_key)
      }
    end
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
    if expected.is_a?(Array) && actual.is_a?(Array)
      return 0.0 if expected.empty? && actual.empty?

      length = [expected.length, actual.length].max
      (0...length).map do |index|
        left = expected[index]
        right = actual[index]
        if left.nil? || right.nil?
          max_abs_value(left.nil? ? right : left)
        else
          max_abs_diff(left, right)
        end
      end.max || 0.0
    elsif expected.is_a?(Hash) && actual.is_a?(Hash)
      keys = expected.keys | actual.keys
      keys.map do |key|
        if !expected.key?(key) || !actual.key?(key)
          max_abs_value(expected[key] || actual[key])
        else
          max_abs_diff(expected.fetch(key), actual.fetch(key))
        end
      end.max || 0.0
    else
      return max_abs_value(actual) if expected.nil?
      return max_abs_value(expected) if actual.nil?

      if expected.is_a?(Array) || expected.is_a?(Hash) || actual.is_a?(Array) || actual.is_a?(Hash)
        return [max_abs_value(expected), max_abs_value(actual)].max
      end

      if (expected == true || expected == false || actual == true || actual == false)
        return expected == actual ? 0.0 : 1.0
      end

      (expected.to_f - actual.to_f).abs
    end
  end

  def max_abs_value(value)
    if value.is_a?(Array)
      return 0.0 if value.empty?

      value.map { |item| max_abs_value(item) }.max || 0.0
    elsif value.is_a?(Hash)
      return 0.0 if value.empty?

      value.values.map { |item| max_abs_value(item) }.max || 0.0
    elsif value == true || value == false
      value ? 1.0 : 0.0
    else
      value.to_f.abs
    end
  end

  def ruby_script_path(spec)
    base = spec.fetch("ruby_script")
    candidate = if @mode == "no_dsl"
      no_dsl = File.join("no_dsl", base)
      File.exist?(File.join(@submodule_root, no_dsl)) ? no_dsl : base
    else
      base
    end
    path = File.join(@submodule_root, candidate)
    raise "Missing examples ruby benchmark script: #{path}" unless File.exist?(path)

    candidate
  end

  def force_device_script
    suffix = @device == "cpu" ? "force_cpu.rb" : "force_gpu.rb"
    File.join(@submodule_root, "benchmark", "ruby", suffix)
  end

  def parse_json_payload(stdout)
    line = stdout.lines.reverse.find { |entry| !entry.strip.empty? }
    raise "Python benchmark did not emit JSON payload" if line.nil?

    JSON.parse(line)
  end

  def parse_benchmark_seconds(stdout)
    line = stdout.lines.reverse.find { |entry| entry.include?("BENCHMARK_SECONDS=") }
    raise "Ruby benchmark did not emit BENCHMARK_SECONDS line" if line.nil?

    raw = line.split("=", 2).last
    raise "Malformed BENCHMARK_SECONDS line" if raw.nil?

    Float(raw.strip)
  end

  def parse_ruby_parity(stdout)
    line = stdout.lines.reverse.find { |entry| entry.include?("BENCHMARK_PARITY=") }
    return nil if line.nil?

    raw = line.split("=", 2).last
    return nil if raw.nil?

    JSON.parse(raw.strip)
  rescue JSON::ParserError
    nil
  end

  def parity_failures_for(parity_payload)
    return [] if parity_payload.nil?

    PARITY_KEYS.reject do |key|
      value = parity_payload[key]
      value = parity_payload[key.to_sym] if value.nil?
      value == true
    end
  end

  def failed_result(kind, command_result)
    {
      status: command_result.fetch(:status),
      seconds: nil,
      samples: [],
      payload: nil,
      parity: nil,
      stderr: "#{kind} command failed:\nstdout:\n#{command_result.fetch(:stdout)}\nstderr:\n#{command_result.fetch(:stderr)}"
    }
  end

  def format_seconds(value)
    return "n/a" if value.nil?

    format("%.6f", value)
  end

  def format_ratio(numerator, denominator)
    return "n/a" if numerator.nil? || denominator.nil? || denominator.zero?

    format("%.2fx", numerator / denominator)
  end

  def format_ms(value)
    return "n/a" if value.nil?

    format("%.3f", value)
  end

  def format_diff(value)
    return "n/a" if value.nil?

    format("%.6f", value)
  end

  def run_command!(command, chdir:, env:, timeout:, label:)
    result = run_capture(command, chdir: chdir, env: env, timeout: timeout)
    return if result.fetch(:status).zero?

    raise "#{label} failed:\nstdout:\n#{result.fetch(:stdout)}\nstderr:\n#{result.fetch(:stderr)}"
  end

  def run_capture(command, chdir:, env:, timeout:)
    stdout = +""
    stderr = +""
    timed_out = false
    status = nil
    started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    Open3.popen3(env, *command, chdir: chdir.to_s) do |stdin, out, err, wait_thr|
      stdin.close
      out_reader = Thread.new { out.read }
      err_reader = Thread.new { err.read }

      if timeout && timeout > 0
        if wait_thr.join(timeout).nil?
          timed_out = true
          begin
            Process.kill("TERM", wait_thr.pid)
          rescue Errno::ESRCH
            nil
          end
          wait_thr.join(5)
          if wait_thr.alive?
            begin
              Process.kill("KILL", wait_thr.pid)
            rescue Errno::ESRCH
              nil
            end
            wait_thr.join
          end
        end
      else
        wait_thr.join
      end

      stdout = out_reader.value
      stderr = err_reader.value
      status = wait_thr.value
    end

    {
      status: timed_out ? 124 : (status&.exitstatus || (status&.success? ? 0 : 1)),
      stdout: stdout,
      stderr: stderr,
      elapsed: Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at,
      timeout: timed_out
    }
  rescue StandardError => e
    {
      status: 1,
      stdout: stdout,
      stderr: "#{e.class}: #{e.message}\n#{Array(e.backtrace).join("\n")}",
      elapsed: Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at,
      timeout: false
    }
  end
end
