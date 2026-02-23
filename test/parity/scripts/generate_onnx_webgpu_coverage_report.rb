#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "rbconfig"
require "stringio"
require "tempfile"

REPO_ROOT = Pathname.new(File.expand_path("../../..", __dir__)).freeze
LIB_ROOT = REPO_ROOT.join("lib").freeze
PARITY_ROOT = REPO_ROOT.join("test", "parity", "reports").freeze
OUT_FILE = PARITY_ROOT.join("ir_webgpu_coverage.json").freeze
SUBMODULE_ROOT = REPO_ROOT.join("submodules", "mlx-ruby-examples").freeze
SUBMODULE_RUNNER = SUBMODULE_ROOT.join("benchmark", "runner.rb").freeze
SUBMODULE_FORCE_CPU = SUBMODULE_ROOT.join("benchmark", "ruby", "force_cpu.rb").freeze
SUBMODULE_CAPTURE_HOOK = REPO_ROOT.join("examples", "benchmark", "benchmark_mlx_examples", "onnx_capture_hook.rb").freeze

$LOAD_PATH.unshift(LIB_ROOT.to_s) unless $LOAD_PATH.include?(LIB_ROOT.to_s)
require "mlx"
require_relative "../../../examples/benchmark/transformer_example"
require_relative "../../../examples/benchmark/cnn_example"
require_relative "../../../examples/benchmark/mlp_example"
require_relative "../../../examples/benchmark/rnn_example"
require_relative "../../../examples/benchmark/gpt2mini_example"

LOCAL_MODELS = %i[transformer cnn mlp rnn karpathy_gpt2].freeze

def benchmark_payload(model_name)
  seed = MLX::Core.array([0.0], MLX::Core.float32)

  trace = case model_name
  when :transformer
    example = BenchmarkExamples::TransformerExample.new(
      batch_size: 1,
      sequence_length: 8,
      target_sequence_length: 4,
      dims: 32,
      num_heads: 4,
      num_layers: 1,
      dtype: MLX::Core.float32
    )
    ->(_seed) { example.run_step }
  when :cnn
    example = BenchmarkExamples::CnnExample.new(
      batch_size: 1,
      dtype: MLX::Core.float32
    )
    ->(_seed) { example.run_step }
  when :mlp
    example = BenchmarkExamples::MlpExample.new(
      batch_size: 1,
      dims: 32,
      dtype: MLX::Core.float32
    )
    ->(_seed) { example.run_step }
  when :rnn
    example = BenchmarkExamples::RnnExample.new(
      batch_size: 1,
      sequence_length: 8,
      dims: 32,
      dtype: MLX::Core.float32
    )
    ->(_seed) { example.run_step }
  when :karpathy_gpt2
    example = BenchmarkExamples::Gpt2MiniExample.new(
      batch_size: 1,
      sequence_length: 16,
      dims: 32,
      num_heads: 4,
      num_layers: 1,
      repo_root: REPO_ROOT.to_s
    )
    model = example.instance_variable_get(:@model)
    sample_input, = example.send(:batch_for_step, 0)
    ->(_seed) { model.call(sample_input) }
  else
    raise ArgumentError, "unknown benchmark model fixture: #{model_name.inspect}"
  end

  JSON.parse(MLX::ONNX.export_graph_ir_json(trace, seed))
end

def graph_report_from_payload(payload)
  JSON.parse(MLX::ONNX::Native.ir_compatibility_report_json(payload))
end

def submodule_available?
  SUBMODULE_RUNNER.exist? &&
    SUBMODULE_FORCE_CPU.exist? &&
    SUBMODULE_CAPTURE_HOOK.exist?
end

def model_filter
  raw = ENV["IR_COVERAGE_MODELS"].to_s
  return nil if raw.empty?

  raw.split(",").map(&:strip).reject(&:empty?).uniq
end

def unbundled_env(overrides = {})
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

def select_local_model_names!(filter)
  names = LOCAL_MODELS.map(&:to_s)
  return names if filter.nil?

  unknown = filter - names
  raise ArgumentError, "unknown local coverage model(s): #{unknown.join(', ')}" unless unknown.empty?

  filter
end

def load_submodule_specs
  specs = []
  current = {}

  File.foreach(SUBMODULE_RUNNER) do |line|
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

  raise "failed to parse model specs from #{SUBMODULE_RUNNER}" if specs.empty?

  specs
end

def select_submodule_specs!(filter)
  specs = load_submodule_specs
  return specs if filter.nil?

  known = specs.map { |spec| spec.fetch("id") }
  unknown = filter - known
  raise ArgumentError, "unknown submodule coverage model(s): #{unknown.join(', ')}" unless unknown.empty?

  specs.select { |spec| filter.include?(spec.fetch("id")) }
end

def synthetic_failure_report(message)
  {
    "format" => "webgpu_compat_report_v1",
    "ir_version" => MLX::ONNX::ONNX_VERSION,
    "total_nodes" => 0,
    "supported_nodes" => 0,
    "unsupported_nodes" => 0,
    "unsupported_ops" => [],
    "ready_for_stub_conversion" => false,
    "nodes" => [],
    "error" => message
  }
end

def run_submodule_coverage(spec)
  Tempfile.create(["ir_webgpu_coverage", ".json"]) do |capture|
    env = unbundled_env(
      "MLX_BENCHMARK" => "1",
      "MLX_BENCHMARK_DEVICE" => "cpu",
      "MLX_BENCHMARK_DRYRUN" => "1",
      "MLX_EXAMPLES_SUBMODULE_ROOT" => SUBMODULE_ROOT.to_s,
      "MLX_EXAMPLES_ONNX_CAPTURE_FILE" => capture.path
    )
    command = [
      RbConfig.ruby,
      "-I#{LIB_ROOT}",
      "-r", SUBMODULE_FORCE_CPU.to_s,
      "-r", SUBMODULE_CAPTURE_HOOK.to_s,
      spec.fetch("ruby_script")
    ]
    stdout, stderr, status = Open3.capture3(env, *command, chdir: SUBMODULE_ROOT.to_s)

    raw = if File.exist?(capture.path) && !File.zero?(capture.path)
      JSON.parse(File.read(capture.path))
    else
      {}
    end

    if status.success? && raw["error"].nil? && raw["payload"].is_a?(Hash)
      report = graph_report_from_payload(raw.fetch("payload"))
      return [report, nil]
    end

    details = []
    details << raw["error"] if raw.is_a?(Hash) && raw["error"].is_a?(String)
    details << "process exited #{status.exitstatus}" unless status.success?
    details << stdout.lines.last.to_s.strip unless stdout.to_s.empty?
    details << stderr.lines.last.to_s.strip unless stderr.to_s.empty?
    message = details.reject(&:empty?).uniq.join(" | ")
    message = "coverage capture failed for #{spec.fetch('id')}" if message.empty?
    [synthetic_failure_report(message), message]
  end
end

def whisper_capture_error_ignorable?(model_id, error)
  model_id == "whisper" &&
    error.to_s.include?("Failed to capture IR payload from benchmark eval outputs")
end

def model_selection
  force_local = ENV["IR_COVERAGE_FORCE_LOCAL"] == "1"
  filter = model_filter
  if !force_local && submodule_available?
    specs = select_submodule_specs!(filter)
    return ["submodule", specs.map { |spec| spec.fetch("id") }, specs]
  end

  model_names = select_local_model_names!(filter)
  local_specs = model_names.map { |name| { "id" => name } }
  ["local", model_names, local_specs]
end

previous_device = MLX::Core.default_device
MLX::Core.set_default_device(MLX::Core.cpu)

unsupported_ops_by_model = {}
unsupported_nodes_by_model = {}
total_nodes_by_model = {}
ready_for_stub_conversion_by_model = {}
errors_by_model = {}

begin
  mode, models, specs = model_selection

  specs.each do |spec|
    key = spec.fetch("id")
    report, error = if mode == "submodule"
      run_submodule_coverage(spec)
    else
      [graph_report_from_payload(benchmark_payload(key.to_sym)), nil]
    end

    unsupported_ops_by_model[key] = report.fetch("unsupported_ops").sort
    unsupported_nodes_by_model[key] = report.fetch("unsupported_nodes")
    total_nodes_by_model[key] = report.fetch("total_nodes")
    ready_for_stub_conversion_by_model[key] = report.fetch("ready_for_stub_conversion")
    if error && !whisper_capture_error_ignorable?(key, error)
      errors_by_model[key] = error
    end
  end

  unsupported_ops_union = unsupported_ops_by_model.values.flatten.uniq.sort

  payload = {
    "format" => "ir_webgpu_coverage_v1",
    "mode" => mode,
    "submodule_present" => submodule_available?,
    "models" => models,
    "model_count" => models.length,
    "unsupported_ops_by_model" => unsupported_ops_by_model,
    "unsupported_nodes_by_model" => unsupported_nodes_by_model,
    "total_nodes_by_model" => total_nodes_by_model,
    "ready_for_stub_conversion_by_model" => ready_for_stub_conversion_by_model,
    "errors_by_model" => errors_by_model,
    "unsupported_ops_union" => unsupported_ops_union
  }

  OUT_FILE.dirname.mkpath
  OUT_FILE.write(JSON.pretty_generate(payload) + "\n")
  puts "wrote #{OUT_FILE}"
ensure
  MLX::Core.set_default_device(previous_device)
end
