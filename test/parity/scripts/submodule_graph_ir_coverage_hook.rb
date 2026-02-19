# frozen_string_literal: true

require "json"
require "set"
require "stringio"
require "mlx"

submodule_root = ENV["MLX_GRAPH_IR_COVERAGE_SUBMODULE_ROOT"].to_s
if submodule_root.empty?
  abort("MLX_GRAPH_IR_COVERAGE_SUBMODULE_ROOT must be set for graph-ir coverage hook")
end

require File.expand_path("benchmark/parity", submodule_root)

module SubmoduleGraphIrCoverageHook
  module_function

  def install!
    return if @installed
    return unless defined?(BenchmarkParity)

    install_eval_hook!

    singleton = BenchmarkParity.singleton_class
    unless singleton.method_defined?(:__graph_ir_coverage_original_validate!)
      singleton.alias_method :__graph_ir_coverage_original_validate!, :validate!
    end

    singleton.define_method(:validate!) do |model_id:, inputs:, outputs:, python_bin: ENV.fetch("PYTHON_BIN", "python3")|
      capture_file = ENV["MLX_GRAPH_IR_COVERAGE_CAPTURE_FILE"].to_s
      if capture_file.empty?
        return __graph_ir_coverage_original_validate!(
          model_id: model_id,
          inputs: inputs,
          outputs: outputs,
          python_bin: python_bin
        )
      end

      report = SubmoduleGraphIrCoverageHook.best_report
      report_error = nil
      if report.nil?
        warn "[graph-ir-hook] no best report at validate for #{model_id}" if SubmoduleGraphIrCoverageHook.debug?
        report, report_error = SubmoduleGraphIrCoverageHook.capture_report(outputs)
        if report
          warn "[graph-ir-hook] fallback capture total_nodes=#{report.fetch('total_nodes', 0)}" if SubmoduleGraphIrCoverageHook.debug?
        else
          warn "[graph-ir-hook] fallback capture error=#{report_error}" if SubmoduleGraphIrCoverageHook.debug?
        end
      elsif SubmoduleGraphIrCoverageHook.debug?
        warn "[graph-ir-hook] using best report total_nodes=#{report.fetch('total_nodes', 0)} for #{model_id}"
      end

      payload = { "model_id" => model_id.to_s }
      if report
        payload["report"] = report
      else
        payload["error"] = report_error || "graph capture failed"
      end
      File.write(capture_file, JSON.generate(payload))
      puts "BENCHMARK_PARITY=#{JSON.generate({ "model" => model_id.to_s, "captured" => true })}" if ENV["MLX_BENCHMARK"] == "1"
      payload
    end

    @installed = true
  end

  def install_eval_hook!
    core_singleton = MLX::Core.singleton_class
    unless core_singleton.method_defined?(:__graph_ir_coverage_original_eval)
      core_singleton.alias_method :__graph_ir_coverage_original_eval, :eval
    end

    core_singleton.define_method(:eval) do |*args|
      SubmoduleGraphIrCoverageHook.observe_eval_values(args) if SubmoduleGraphIrCoverageHook.capture_enabled?
      __graph_ir_coverage_original_eval(*args)
    end
  end

  def capture_enabled?
    !ENV["MLX_GRAPH_IR_COVERAGE_CAPTURE_FILE"].to_s.empty?
  end

  def debug?
    ENV["MLX_GRAPH_IR_COVERAGE_DEBUG"] == "1"
  end

  def observe_eval_values(values)
    if debug?
      source = caller(2, 1).first
      arg_types = values.map { |value| value.class.name rescue value.class.to_s }
      warn "[graph-ir-hook] eval hook called with #{values.length} arg(s) from #{source} types=#{arg_types.join(',')}"
    end
    report, error = capture_report(values)
    if report.nil?
      warn "[graph-ir-hook] eval capture error=#{error}" if debug? && error
      return
    end

    total_nodes = report.fetch("total_nodes", 0).to_i
    warn "[graph-ir-hook] observed eval total_nodes=#{total_nodes}" if debug?
    if total_nodes > @best_total_nodes.to_i
      @best_total_nodes = total_nodes
      @best_report = report
      warn "[graph-ir-hook] updated best total_nodes=#{total_nodes}" if debug?
    end
  rescue StandardError
    nil
  end

  def best_report
    @best_report
  end

  def capture_report(outputs)
    arrays = collect_mlx_arrays(outputs)
    return [nil, "no MLX array outputs found"] if arrays.empty?

    dot_io = StringIO.new
    MLX::Core.export_to_dot(dot_io, *arrays)
    dot_graph = dot_io.string
    if debug? && !dot_graph.include?("label =")
      warn "[graph-ir-hook] dot graph had no op labels:\n#{dot_graph}"
    end
    graph_payload = dot_to_graph_ir_payload(dot_graph)
    [MLX::Core.graph_ir_webgpu_compatibility_report(graph_payload), nil]
  rescue StandardError => e
    [nil, "#{e.class}: #{e.message}"]
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

  def mlx_array_like?(value)
    value.respond_to?(:shape) && value.respond_to?(:dtype) && value.respond_to?(:to_a) && !value.is_a?(::Array)
  end

  def dot_to_graph_ir_payload(dot_graph)
    op_order, op_by_node, edges = parse_dot(dot_graph)
    raise ArgumentError, "dot graph did not contain operator nodes" if op_order.empty?

    node_inputs = Hash.new { |hash, key| hash[key] = [] }
    node_outputs = Hash.new { |hash, key| hash[key] = [] }

    edges.each do |from, to|
      if op_by_node.key?(to) && !op_by_node.key?(from)
        node_inputs[to] << from
      elsif op_by_node.key?(from) && !op_by_node.key?(to)
        node_outputs[from] << to
      end
    end

    nodes = op_order.each_with_index.map do |node_id, index|
      inputs = ordered_unique(node_inputs.fetch(node_id, []))
      outputs = ordered_unique(node_outputs.fetch(node_id, []))
      inputs = ["hook_input_#{index}"] if inputs.empty?
      outputs = ["hook_output_#{index}"] if outputs.empty?
      {
        "op" => op_by_node.fetch(node_id),
        "inputs" => inputs,
        "outputs" => outputs
      }
    end

    produced = Set.new(nodes.flat_map { |node| node.fetch("outputs") })
    consumed = Set.new(nodes.flat_map { |node| node.fetch("inputs") })

    input_names = ordered_unique(nodes.flat_map { |node| node.fetch("inputs") }.reject { |name| produced.include?(name) })
    output_names = ordered_unique(nodes.flat_map { |node| node.fetch("outputs") }.reject { |name| consumed.include?(name) })
    output_names = [nodes.last.fetch("outputs").first] if output_names.empty?

    {
      "ir_version" => MLX::GraphIR::IR_VERSION,
      "shapeless" => false,
      "inputs" => input_names.map { |name| tensor_info(name) },
      "keyword_inputs" => [],
      "outputs" => output_names.map { |name| tensor_info(name) },
      "constants" => [],
      "nodes" => nodes
    }
  end

  def tensor_info(name)
    {
      "name" => name,
      "shape" => [1],
      "dtype" => "float32"
    }
  end

  def parse_dot(dot_graph)
    op_order = []
    op_by_node = {}
    edges = []

    dot_graph.each_line do |line|
      label_match = line.match(/\{\s*([A-Za-z0-9_]+)\s+\[label\s*=\s*"([^"]+)"/)
      if label_match
        node_id = label_match[1]
        op_by_node[node_id] = label_match[2]
        op_order << node_id
        next
      end

      next unless line.include?("->")

      from_raw, to_raw = line.split("->", 2)
      from = normalize_dot_token(from_raw)
      to = normalize_dot_token(to_raw)
      next if from.empty? || to.empty?

      edges << [from, to]
    end

    [op_order, op_by_node, edges]
  end

  def normalize_dot_token(raw)
    token = raw.to_s.strip
    token = token.sub(/\A\{\s*/, "").sub(/\s*\}\z/, "")
    token = token.sub(/;+\z/, "").strip
    token = token.sub(/\A"/, "").sub(/"\z/, "")
    token.strip
  end

  def ordered_unique(items)
    seen = Set.new
    items.each_with_object([]) do |item, out|
      next if seen.include?(item)

      seen.add(item)
      out << item
    end
  end
end

SubmoduleGraphIrCoverageHook.install!
