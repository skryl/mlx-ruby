# frozen_string_literal: true

require "fileutils"
require "json"
require "stringio"

REPO_ROOT = File.expand_path("../..", __dir__)
LIB_ROOT = File.join(REPO_ROOT, "lib")
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)

require "mlx"
require_relative "../../examples/web/gpt2_example"

module Gpt2WebAssets
  module_function

  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "gpt2")
  WEIGHTS_DIR = File.join(OUTPUT_DIR, "weights")
  WEIGHTS_REPO_MARKER = File.join(WEIGHTS_DIR, ".repo_id")
  MODEL_NAME = "gpt2_ruby_web_demo"
  DEFAULT_CONTEXT_SIZE = 64
  DEFAULT_EOS_TOKEN_ID = 50_256
  TOKENIZER_REPO_MARKER = File.join(OUTPUT_DIR, ".tokenizer_repo")

  DEFAULT_HF_REPO_ID = "openai-community/gpt2"
  DEFAULT_MODEL_DTYPE = "float32"
  REQUIRED_TOKENIZER_FILES = %w[vocab.json merges.txt tokenizer.json tokenizer_config.json].freeze
  OPTIONAL_TOKENIZER_FILES = %w[generation_config.json].freeze

  PROMPT_PRESETS = {
    "once upon a time" => "Once upon a time, in a city lit by neon rain,",
    "the future" => "The future of software is",
    "ruby and webgpu" => "Ruby, ONNX Runtime WebGPU, and MLX together can",
    "shakespeare remix" => "To be, or not to be, that is"
  }.freeze

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    repo_id = ENV.fetch("GPT2_HF_REPO", DEFAULT_HF_REPO_ID)
    model_dtype = resolve_model_dtype(ENV.fetch("GPT2_MODEL_DTYPE", DEFAULT_MODEL_DTYPE))
    FileUtils.mkdir_p(OUTPUT_DIR)
    FileUtils.mkdir_p(WEIGHTS_DIR)
    cleanup_legacy_hf_model_dir!

    tokenizer_repo_changed = current_tokenizer_repo_id != repo_id
    fetch_tokenizer_files!(repo_id, force: tokenizer_repo_changed)

    weights_repo_changed = current_weights_repo_id != repo_id
    clear_weights_dir! if weights_repo_changed

    hf_model_dir = WEIGHTS_DIR
    Gpt2Example::Gpt2Model.download_hf_weights!(hf_model_dir, repo_id: repo_id)

    config_payload = JSON.parse(File.binread(File.join(hf_model_dir, "config.json")))
    generation_config = read_json_or_default(
      File.join(OUTPUT_DIR, "generation_config.json"),
      {}
    )
    config = Gpt2Example::Config.from_hf_config(config_payload)

    context_size = resolve_context_size(config.n_positions)
    eos_token_id = Integer(generation_config.fetch("eos_token_id", DEFAULT_EOS_TOKEN_ID))
    vocab = JSON.parse(File.binread(File.join(OUTPUT_DIR, "vocab.json")))

    model = Gpt2Example::Gpt2Model.from_hf_directory(hf_model_dir, dtype: model_dtype)
    model.eval
    MLX::Core.eval(model.parameters)

    seed_tokens = Array.new(context_size, eos_token_id)
    input_seed = MLX::Core.array([seed_tokens], MLX::Core.int32)
    graph_ir_path = File.join(OUTPUT_DIR, "graph_ir.json")
    MLX::Core.export_graph_ir(
      graph_ir_path,
      ->(input_ids) { model.call(input_ids) },
      input_seed
    )
    payload = JSON.parse(read_file_with_fallback(graph_ir_path))

    onnx_path = File.join(OUTPUT_DIR, "model.onnx")
    MLX::Core.export_onnx(onnx_path, payload, model_name: MODEL_NAME)

    inputs = normalize_io_specs(payload.fetch("inputs"))
    outputs = normalize_io_specs(payload.fetch("outputs"))

    metadata = {
      "format" => "gpt2_ruby_demo_asset_v1",
      "model_name" => MODEL_NAME,
      "source" => "Gpt2Example::Gpt2Model (HF weights from #{repo_id})",
      "input" => inputs.first,
      "output" => outputs.first,
      "inputs" => inputs,
      "outputs" => outputs,
      "config" => config_payload,
      "generation_config" => generation_config,
      "tokenizer" => {
        "type" => "gpt2_bpe",
        "vocab_size" => Integer(vocab.length),
        "eos_token_id" => eos_token_id
      },
      "generation" => {
        "context_size" => context_size,
        "default_max_tokens" => 80,
        "default_temperature" => 0.8
      },
      "weights" => {
        "source" => "huggingface",
        "repo_id" => repo_id,
        "dtype" => dtype_name(model_dtype),
        "trained" => true,
        "load_path" => File.join(hf_model_dir, "model.safetensors")
      }
    }

    File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
    File.binwrite(File.join(OUTPUT_DIR, "prompt.presets.json"), JSON.pretty_generate(PROMPT_PRESETS))
    FileUtils.cp(File.join(hf_model_dir, "config.json"), File.join(OUTPUT_DIR, "config.json"))
    File.binwrite(TOKENIZER_REPO_MARKER, "#{repo_id}\n")
    File.binwrite(WEIGHTS_REPO_MARKER, "#{repo_id}\n")

    puts "Wrote GPT-2 demo assets to #{OUTPUT_DIR}"
    puts "  repo: #{repo_id}"
    puts "  dtype: #{dtype_name(model_dtype)}"
    puts "  - #{graph_ir_path}"
    puts "  - #{onnx_path}"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
    puts "  - #{File.join(OUTPUT_DIR, 'prompt.presets.json')}"
    puts "  - #{File.join(WEIGHTS_DIR, 'model.safetensors')}"
  end

  def fetch_tokenizer_files!(repo_id, force: false)
    REQUIRED_TOKENIZER_FILES.each do |relative_path|
      download_repo_file!(repo_id: repo_id, relative_path: relative_path, required: true, force: force)
    end
    OPTIONAL_TOKENIZER_FILES.each do |relative_path|
      download_repo_file!(repo_id: repo_id, relative_path: relative_path, required: false, force: force)
    end
  end

  def download_repo_file!(repo_id:, relative_path:, required:, force:)
    url = "https://huggingface.co/#{repo_id}/resolve/main/#{relative_path}"
    destination = File.join(OUTPUT_DIR, relative_path)
    if !force && File.exist?(destination) && File.size(destination).positive?
      return
    end
    File.delete(destination) if force && File.exist?(destination)

    puts "Downloading #{url}"
    system("curl", "-fL", "--retry", "3", "--retry-delay", "2", "-o", destination, url)
    return if $?.success?
    File.delete(destination) if File.exist?(destination)
    return unless required

    raise "Failed downloading #{url} to #{destination}"
  end

  def current_tokenizer_repo_id
    return nil unless File.exist?(TOKENIZER_REPO_MARKER)

    value = File.binread(TOKENIZER_REPO_MARKER).strip
    value.empty? ? nil : value
  end

  def current_weights_repo_id
    return nil unless File.exist?(WEIGHTS_REPO_MARKER)

    value = File.binread(WEIGHTS_REPO_MARKER).strip
    value.empty? ? nil : value
  end

  def clear_weights_dir!
    Dir.glob(File.join(WEIGHTS_DIR, "*")).each do |entry|
      FileUtils.rm_rf(entry)
    end
    File.delete(WEIGHTS_REPO_MARKER) if File.exist?(WEIGHTS_REPO_MARKER)
  end

  def cleanup_legacy_hf_model_dir!
    legacy_dir = File.join(OUTPUT_DIR, "hf_model")
    return unless Dir.exist?(legacy_dir)

    FileUtils.rm_rf(legacy_dir)
  end

  def resolve_model_dtype(value)
    token = value.to_s.strip.downcase
    case token
    when "float16", "f16", "fp16"
      MLX::Core.float16
    when "bfloat16", "bf16"
      MLX::Core.bfloat16
    when "float32", "f32", "fp32", ""
      MLX::Core.float32
    else
      raise ArgumentError, "Unsupported GPT2_MODEL_DTYPE=#{value.inspect}; expected float32|float16|bfloat16"
    end
  end

  def dtype_name(dtype)
    if dtype.respond_to?(:name)
      dtype.name.to_s
    else
      dtype.to_s
    end
  end

  def resolve_context_size(max_positions)
    requested = ENV.fetch("GPT2_CONTEXT", DEFAULT_CONTEXT_SIZE.to_s).to_i
    requested = DEFAULT_CONTEXT_SIZE if requested <= 0
    [[requested, 1].max, Integer(max_positions)].min
  end

  def normalize_io_specs(entries)
    entries.map do |entry|
      {
        "name" => entry.fetch("name"),
        "type" => entry["dtype"] || entry["type"] || "unknown",
        "shape" => entry["shape"]
      }
    end
  end

  def read_json_or_default(path, fallback)
    return fallback unless File.exist?(path)

    JSON.parse(File.binread(path))
  end

  def read_file_with_fallback(path)
    File.binread(path)
  rescue Errno::EINVAL
    File.open(path, "rb") do |io|
      content = +""
      while (chunk = io.read(8 * 1024 * 1024))
        content << chunk
      end
      content
    end
  end
end

if $PROGRAM_NAME == __FILE__
  Gpt2WebAssets.run!
end
