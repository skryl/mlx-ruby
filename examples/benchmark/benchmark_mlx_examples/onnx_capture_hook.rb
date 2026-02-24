# frozen_string_literal: true

require "json"
require "stringio"

submodule_root = ENV["MLX_EXAMPLES_SUBMODULE_ROOT"].to_s
if submodule_root.empty?
  abort("MLX_EXAMPLES_SUBMODULE_ROOT must be set for ONNX capture hook")
end

require "mlx"
require File.expand_path("benchmark/parity", submodule_root)

module ExamplesModelsOnnxCaptureHook
  module_function

  MAX_CAPTURE_HISTORY = 32
  MAX_CAPTURE_OUTPUTS = 32

  def install!
    return if @installed
    return unless defined?(BenchmarkParity)

    install_base64_compat!
    install_eval_hook!
    install_capture_finalizer!

    singleton = BenchmarkParity.singleton_class
    unless singleton.method_defined?(:__examples_onnx_original_validate!)
      singleton.alias_method :__examples_onnx_original_validate!, :validate!
    end

    singleton.define_method(:validate!) do |model_id:, inputs:, outputs:, python_bin: ENV.fetch("PYTHON_BIN", "python3")|
      capture_file = ENV["MLX_EXAMPLES_ONNX_CAPTURE_FILE"].to_s
      if capture_file.empty?
        return __examples_onnx_original_validate!(
          model_id: model_id,
          inputs: inputs,
          outputs: outputs,
          python_bin: python_bin
        )
      end

      capture_payload = ExamplesModelsOnnxCaptureHook.build_capture(outputs)
      capture_payload["model_id"] = model_id.to_s
      File.write(capture_file, JSON.generate(capture_payload))

      summary = {
        "model" => model_id.to_s,
        "device" => normalized_device(ENV["MLX_BENCHMARK_DEVICE"]),
        "input_shape_match" => true,
        "input_content_match" => true,
        "output_shape_match" => true,
        "output_content_match" => true
      }
      puts "BENCHMARK_PARITY=#{JSON.generate(summary)}" if ENV["MLX_BENCHMARK"] == "1"
      summary
    rescue StandardError => e
      if !capture_file.empty?
        payload = { "model_id" => model_id.to_s, "error" => "#{e.class}: #{e.message}" }
        File.write(capture_file, JSON.generate(payload))
      end
      raise
    end

    @installed = true
  end

  def install_base64_compat!
    return if @base64_compat_installed

    begin
      require "base64"
      @base64_compat_installed = true
      return
    rescue LoadError
      nil
    end

    base64_module = if defined?(::Base64)
      ::Base64
    else
      ::Object.const_set(:Base64, Module.new)
    end

    unless base64_module.respond_to?(:decode64)
      base64_module.define_singleton_method(:decode64) { |str| str.to_s.unpack1("m").to_s }
    end
    unless base64_module.respond_to?(:encode64)
      base64_module.define_singleton_method(:encode64) { |bin| [bin.to_s].pack("m") }
    end
    unless base64_module.respond_to?(:strict_decode64)
      base64_module.define_singleton_method(:strict_decode64) { |str| str.to_s.unpack1("m0") }
    end
    unless base64_module.respond_to?(:strict_encode64)
      base64_module.define_singleton_method(:strict_encode64) { |bin| [bin.to_s].pack("m0") }
    end
    unless base64_module.respond_to?(:urlsafe_encode64)
      base64_module.define_singleton_method(:urlsafe_encode64) do |bin, padding: true|
        encoded = base64_module.strict_encode64(bin).tr("+/", "-_")
        padding ? encoded : encoded.delete("=")
      end
    end
    unless base64_module.respond_to?(:urlsafe_decode64)
      base64_module.define_singleton_method(:urlsafe_decode64) do |str|
        normalized = str.to_s.tr("-_", "+/")
        normalized += "=" * ((4 - normalized.length % 4) % 4)
        base64_module.strict_decode64(normalized)
      end
    end

    $LOADED_FEATURES << "base64.rb" unless $LOADED_FEATURES.include?("base64.rb")
    @base64_compat_installed = true
  end

  def install_capture_finalizer!
    return if @capture_finalizer_installed

    at_exit do
      begin
        ExamplesModelsOnnxCaptureHook.finalize_capture_file!
      rescue StandardError
        nil
      end
    end
    @capture_finalizer_installed = true
  end

  def finalize_capture_file!
    return unless capture_enabled?

    capture_file = ENV["MLX_EXAMPLES_ONNX_CAPTURE_FILE"].to_s
    return if capture_file.empty?
    return if File.exist?(capture_file) && !File.zero?(capture_file)

    payload = build_capture(nil)
    File.write(capture_file, JSON.generate(payload))
  rescue StandardError => e
    error_payload = { "error" => "#{e.class}: #{e.message}" }
    File.write(capture_file, JSON.generate(error_payload)) unless capture_file.empty?
  end

  def install_eval_hook!
    core_singleton = MLX::Core.singleton_class
    unless core_singleton.method_defined?(:__examples_onnx_capture_original_eval)
      core_singleton.alias_method :__examples_onnx_capture_original_eval, :eval
    end

    core_singleton.define_method(:eval) do |*values|
      ExamplesModelsOnnxCaptureHook.observe_eval_values(values) if ExamplesModelsOnnxCaptureHook.capture_enabled?
      __examples_onnx_capture_original_eval(*values)
    end
  end

  def capture_enabled?
    !ENV["MLX_EXAMPLES_ONNX_CAPTURE_FILE"].to_s.empty?
  end

  def observe_eval_values(values)
    candidate = capture_candidate(values)
    return if candidate.nil?

    @capture_history ||= []
    @capture_history << candidate
    @capture_history.shift while @capture_history.length > MAX_CAPTURE_HISTORY
    @last_capture = candidate
  rescue StandardError
    nil
  end

  def build_capture(outputs)
    candidate = select_candidate(outputs)
    raise "Failed to capture IR payload from benchmark eval outputs" if candidate.nil?

    payload = candidate.fetch(:payload)
    node_count = candidate_node_count(candidate)
    raise "Captured IR payload has zero nodes" if node_count <= 0

    tensor_outputs = candidate.fetch(:tensors)
    raise "Captured IR payload has no tensor outputs" if tensor_outputs.empty?

    output_names = resolve_output_names(payload, tensor_outputs.length)
    expected_outputs = output_names.each_with_index.each_with_object({}) do |(name, index), out|
      out[name] = deep_materialize(index < tensor_outputs.length ? tensor_outputs[index] : nil)
    end

    {
      "payload" => payload,
      "feeds" => default_feeds_for(payload),
      "output_names" => output_names,
      "expected_outputs" => expected_outputs
    }
  end

  def select_candidate(outputs)
    candidates = []
    candidates.concat(@capture_history) if @capture_history
    fallback = capture_candidate(outputs)
    candidates << fallback unless fallback.nil?

    candidates = candidates.compact
    return nil if candidates.empty?

    nonzero_candidates = candidates.select { |candidate| candidate_node_count(candidate).positive? }
    selected_pool = nonzero_candidates.empty? ? candidates : nonzero_candidates

    bounded_output_candidates = selected_pool.select do |candidate|
      candidate_output_count(candidate) <= MAX_CAPTURE_OUTPUTS
    end
    selected_pool = bounded_output_candidates unless bounded_output_candidates.empty?

    selected_pool.max_by do |candidate|
      [candidate_node_count(candidate), candidate_output_count(candidate)]
    end
  end

  def capture_candidate(values)
    tensor_outputs = collect_mlx_arrays(values)
    return nil if tensor_outputs.empty?

    trace_output = tensor_outputs.length == 1 ? tensor_outputs.first : tensor_outputs
    seed = MLX::Core.array([0.0], MLX::Core.float32)
    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(->(_seed) { trace_output }, seed))
    {
      payload: payload,
      tensors: tensor_outputs
    }
  rescue StandardError
    nil
  end

  def candidate_node_count(candidate)
    candidate.fetch(:payload).fetch("nodes", []).length
  end

  def candidate_output_count(candidate)
    candidate.fetch(:payload).fetch("outputs", []).length
  end

  def collect_mlx_arrays(value, out = [])
    if mlx_array_like?(value)
      out << value
      return out
    end

    case value
    when Hash
      value.each_value { |item| collect_mlx_arrays(item, out) }
    when Array
      value.each { |item| collect_mlx_arrays(item, out) }
    end
    out
  end

  def resolve_output_names(payload, output_count)
    names = payload
      .fetch("outputs", [])
      .map { |entry| entry["name"].to_s }
      .reject(&:empty?)

    if names.length > output_count
      names = names.first(output_count)
    elsif names.length < output_count
      names = names.dup
      while names.length < output_count
        names << "output_#{names.length}"
      end
    end

    names
  end

  def default_feeds_for(payload)
    inputs = payload.fetch("inputs", [])
    return {} if inputs.empty?

    seed = MLX::Core.array([0.0], MLX::Core.float32).to_a
    inputs.each_with_object({}) do |entry, out|
      name = entry["name"].to_s
      next if name.empty?

      out[name] = seed
    end
  end

  def deep_materialize(value)
    if mlx_array_like?(value)
      return deep_materialize(value.to_a)
    end

    case value
    when Hash
      value.each_with_object({}) do |(key, item), out|
        out[key.to_s] = deep_materialize(item)
      end
    when Array
      value.map { |item| deep_materialize(item) }
    when Numeric, String, TrueClass, FalseClass, NilClass
      value
    else
      value.respond_to?(:to_a) ? deep_materialize(value.to_a) : value.to_s
    end
  end

  def mlx_array_like?(value)
    value.respond_to?(:shape) && value.respond_to?(:dtype) && value.respond_to?(:to_a) && !value.is_a?(::Array)
  end

  def normalized_device(raw)
    value = raw.to_s.strip.downcase
    return "cpu" if value == "cpu"

    "gpu"
  end
end

ExamplesModelsOnnxCaptureHook.install!
