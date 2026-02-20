# frozen_string_literal: true

require "fileutils"
require "json"
require "stringio"

REPO_ROOT = File.expand_path("../..", __dir__)
LIB_ROOT = File.join(REPO_ROOT, "lib")
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)

require "mlx"
require_relative "../../examples/web/nanogpt_example"

module NanoGptWebAssets
  module_function

  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "nanogpt")
  MODEL_NAME = "nanogpt_shakespeare_web_demo"
  WEIGHTS_PATH = File.join(OUTPUT_DIR, "weights.npz")
  TOKENIZER_PATH = File.join(OUTPUT_DIR, "tokenizer.json")
  TRAINING_CONFIG_PATH = File.join(OUTPUT_DIR, "training_config.json")
  GENERATED_EXPORT_FILES = %w[graph_ir.json model.onnx meta.json prompt.presets.json].freeze

  DEFAULT_PROMPT_PRESETS = {
    "to be" => "To be, or not to be",
    "romeo" => "Romeo, Romeo, wherefore art thou Romeo?",
    "king" => "KING HENRY:",
    "fools" => "The fool doth think he is wise,"
  }.freeze

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    artifacts = load_training_artifacts
    unless artifacts
      handle_missing_training_artifacts!
      return
    end

    tokenizer = artifacts.fetch(:tokenizer)
    model_config = symbolize_keys(artifacts.fetch(:training_config).fetch("model"))
    model_config[:vocab_size] = Integer(tokenizer.fetch("vocab_size"))
    model_config[:block_size] = Integer(model_config.fetch(:block_size))

    seed = Integer(artifacts.fetch(:training_config).fetch("seed"))
    presets = artifacts.fetch(:training_config)["prompt_presets"]
    prompt_presets = presets.is_a?(Hash) && !presets.empty? ? presets : DEFAULT_PROMPT_PRESETS

    MLX::Core.random_seed(seed)
    model = NanoGptExample::NanoGptModel.new(**model_config)
    model.load_weights(WEIGHTS_PATH)
    model.eval
    MLX::Core.eval(model.parameters)

    pad_id = Integer(tokenizer.fetch("pad_id", 0))
    seed_prompt = prompt_presets.values.first.to_s
    input_seed = prompt_to_seed_input(
      prompt: seed_prompt,
      tokenizer: tokenizer,
      block_size: model_config.fetch(:block_size),
      pad_id: pad_id
    )

    payload_json = MLX::GraphIR.export_graph_ir(
      StringIO.new,
      ->(tokens) { model.call(tokens) },
      input_seed
    )
    payload = JSON.parse(payload_json)
    report = MLX::GraphIR.webgpu_compatibility_report(payload)
    unsupported_nodes = report.fetch("unsupported_nodes").to_i
    unless unsupported_nodes.zero?
      abort(
        "nanoGPT graph is not fully WebGPU-compatible. " \
        "unsupported_ops=#{report.fetch('unsupported_ops').inspect}"
      )
    end

    FileUtils.mkdir_p(OUTPUT_DIR)

    graph_ir_path = File.join(OUTPUT_DIR, "graph_ir.json")
    File.binwrite(graph_ir_path, JSON.pretty_generate(payload))

    onnx_path = File.join(OUTPUT_DIR, "model.onnx")
    MLX::GraphIR.export_onnx(onnx_path, payload, model_name: MODEL_NAME)

    metadata = {
      "format" => "nanogpt_shakespeare_demo_asset_v1",
      "model_name" => MODEL_NAME,
      "seed" => seed,
      "trained" => true,
      "dataset" => "shakespeare_char",
      "input" => payload.fetch("inputs").first,
      "output" => payload.fetch("outputs").first,
      "tokenizer" => tokenizer,
      "model" => stringify_keys(model_config),
      "generation" => {
        "context_size" => model_config.fetch(:block_size),
        "default_max_tokens" => 200,
        "default_temperature" => 0.8,
        "pad_id" => pad_id
      },
      "training" => artifacts.fetch(:training_config)["training"],
      "webgpu_compatibility" => {
        "unsupported_nodes" => unsupported_nodes,
        "unsupported_ops" => report.fetch("unsupported_ops")
      }
    }

    File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
    File.binwrite(File.join(OUTPUT_DIR, "prompt.presets.json"), JSON.pretty_generate(prompt_presets))

    puts "Wrote nanoGPT demo assets to #{OUTPUT_DIR}"
    puts "  mode: trained checkpoint"
    puts "  - #{graph_ir_path}"
    puts "  - #{onnx_path}"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
    puts "  - #{File.join(OUTPUT_DIR, 'prompt.presets.json')}"
  end

  def prompt_to_seed_input(prompt:, tokenizer:, block_size:, pad_id:)
    prompt_ids = []
    mapping = tokenizer.fetch("char_to_id")
    prompt.each_char do |char|
      encoded = mapping[char]
      prompt_ids << Integer(encoded) unless encoded.nil?
    end
    trimmed = prompt_ids.last(block_size)
    padded = if trimmed.length < block_size
      Array.new(block_size - trimmed.length, pad_id) + trimmed
    else
      trimmed
    end
    MLX::Core.array([padded], MLX::Core.int32)
  end

  def stringify_keys(hash)
    hash.each_with_object({}) { |(key, value), out| out[key.to_s] = value }
  end

  def symbolize_keys(hash)
    hash.each_with_object({}) { |(key, value), out| out[key.to_sym] = value }
  end

  def load_training_artifacts
    required = [WEIGHTS_PATH, TOKENIZER_PATH, TRAINING_CONFIG_PATH]
    return nil unless required.all? { |path| File.exist?(path) }

    {
      tokenizer: JSON.parse(File.binread(TOKENIZER_PATH)),
      training_config: JSON.parse(File.binread(TRAINING_CONFIG_PATH))
    }
  end

  def handle_missing_training_artifacts!
    clear_generated_assets!
    puts "Skipping nanoGPT demo export: missing trained artifacts."
    puts "  required files:"
    puts "  - #{WEIGHTS_PATH}"
    puts "  - #{TOKENIZER_PATH}"
    puts "  - #{TRAINING_CONFIG_PATH}"
  end

  def clear_generated_assets!
    GENERATED_EXPORT_FILES.each do |relative|
      FileUtils.rm_f(File.join(OUTPUT_DIR, relative))
    end
  end
end

if $PROGRAM_NAME == __FILE__
  NanoGptWebAssets.run!
end
