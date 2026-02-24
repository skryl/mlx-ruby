# frozen_string_literal: true

require "fileutils"
require "json"

REPO_ROOT = File.expand_path("../..", __dir__)
LIB_ROOT = File.join(REPO_ROOT, "lib")
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)

require "mlx"
require_relative "../../examples/web/nanogpt_example"

module NanoGptWebAssets
  module_function

  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "nanogpt")
  WEIGHTS_DIR = File.join(OUTPUT_DIR, "weights")
  WEIGHTS_REPO_MARKER = File.join(WEIGHTS_DIR, ".repo_id")
  MODEL_NAME = "nanogpt_shakespeare_web_demo"
  DEFAULT_HF_REPO_ID = NanoGptExample::HF_REPO_ID
  DEFAULT_MODEL_DTYPE = "float32"
  DEFAULT_SEED = 1337

  PROMPT_PRESETS = {
    "to be" => "To be, or not to be",
    "romeo" => "Romeo, Romeo, wherefore art thou Romeo?",
    "king" => "KING HENRY:",
    "fools" => "The fool doth think he is wise,"
  }.freeze

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    timings = {}
    repo_id = ENV.fetch("NANOGPT_HF_REPO", DEFAULT_HF_REPO_ID)
    model_dtype = resolve_model_dtype(ENV.fetch("NANOGPT_MODEL_DTYPE", DEFAULT_MODEL_DTYPE))
    seed = resolve_seed(ENV["NANOGPT_SEED"])
    dataset_path = resolve_dataset_path(ENV["NANOGPT_DATASET"])

    FileUtils.mkdir_p(OUTPUT_DIR)
    FileUtils.mkdir_p(WEIGHTS_DIR)

    benchmark_step(timings, "fetch_model_weights") do
      weights_repo_changed = current_weights_repo_id != repo_id
      clear_weights_dir! if weights_repo_changed
      NanoGptExample::NanoGptModel.download_hf_weights!(WEIGHTS_DIR, repo_id: repo_id)
    end

    config_payload = JSON.parse(File.binread(File.join(WEIGHTS_DIR, "config.json")))
    config = NanoGptExample::Config.from_hf_config(config_payload)

    tokenizer = benchmark_step(timings, "build_tokenizer") do
      build_tokenizer_for_dataset!(dataset_path: dataset_path, expected_vocab_size: config.vocab_size)
    end

    MLX::Core.random_seed(seed)
    model = benchmark_step(timings, "load_model_weights_into_mlx") do
      loaded = NanoGptExample::NanoGptModel.from_hf_directory(WEIGHTS_DIR, dtype: model_dtype)
      loaded.eval
      MLX::Core.eval(loaded.parameters)
      loaded
    end

    pad_id = Integer(tokenizer.fetch("pad_id", 0))
    seed_prompt = PROMPT_PRESETS.values.first.to_s
    input_seed = prompt_to_seed_input(
      prompt: seed_prompt,
      tokenizer: tokenizer,
      block_size: config.block_size,
      pad_id: pad_id
    )

    onnx_path = File.join(OUTPUT_DIR, "model.onnx")
    benchmark_step(timings, "export_onnx_binary") do
      MLX::ONNX.export_onnx(
        onnx_path,
        ->(tokens) { model.call(tokens) },
        input_seed,
        model_name: MODEL_NAME
      )
    end

    onnx_io = benchmark_step(timings, "inspect_onnx_io") do
      derive_onnx_io!(input_seed: input_seed, model: model)
    end

    model_config = {
      vocab_size: config.vocab_size,
      block_size: config.block_size,
      n_layer: config.n_layer,
      n_head: config.n_head,
      n_embd: config.n_embd,
      dropout: config.dropout,
      bias: config.bias
    }

    metadata = {
      "format" => "nanogpt_shakespeare_demo_asset_v1",
      "model_name" => MODEL_NAME,
      "seed" => seed,
      "source" => "NanoGptExample::NanoGptModel (HF weights from #{repo_id})",
      "trained" => true,
      "dataset" => "shakespeare_char",
      "dataset_path" => dataset_path,
      "input" => onnx_io.fetch("inputs").first,
      "output" => onnx_io.fetch("outputs").first,
      "tokenizer" => tokenizer,
      "model" => stringify_keys(model_config),
      "config" => config_payload,
      "generation" => {
        "context_size" => config.block_size,
        "default_max_tokens" => 200,
        "default_temperature" => 0.8,
        "pad_id" => pad_id
      },
      "parameters" => {
        "total" => parameter_count(model.parameters)
      },
      "weights" => {
        "source" => "huggingface",
        "repo_id" => repo_id,
        "dtype" => dtype_name(model_dtype),
        "trained" => true,
        "load_path" => File.join(WEIGHTS_DIR, "model.safetensors")
      },
      "webgpu_compatibility" => {
        "unsupported_nodes" => 0,
        "unsupported_ops" => []
      }
    }

    benchmark_step(timings, "write_metadata_and_markers") do
      File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
      File.binwrite(File.join(OUTPUT_DIR, "prompt.presets.json"), JSON.pretty_generate(PROMPT_PRESETS))
      File.binwrite(File.join(OUTPUT_DIR, "tokenizer.json"), JSON.pretty_generate(tokenizer))
      FileUtils.cp(File.join(WEIGHTS_DIR, "config.json"), File.join(OUTPUT_DIR, "config.json"))
      File.binwrite(WEIGHTS_REPO_MARKER, "#{repo_id}\n")
    end

    puts "Wrote nanoGPT demo assets to #{OUTPUT_DIR}"
    puts "  repo: #{repo_id}"
    puts "  dtype: #{dtype_name(model_dtype)}"
    puts "  - #{onnx_path}"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
    puts "  - #{File.join(OUTPUT_DIR, 'tokenizer.json')}"
    puts "  - #{File.join(OUTPUT_DIR, 'prompt.presets.json')}"
    puts "  - #{File.join(WEIGHTS_DIR, 'model.safetensors')}"
    print_timings(timings)
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

  def build_tokenizer_for_dataset!(dataset_path:, expected_vocab_size:)
    text = NanoGptExample::Dataset.load_text(dataset_path)
    tokenizer = NanoGptExample::Dataset.build_char_tokenizer(text)
    actual_vocab_size = Integer(tokenizer.fetch("vocab_size"))
    return tokenizer if actual_vocab_size == Integer(expected_vocab_size)

    raise(
      "Tokenizer vocab mismatch for #{dataset_path}: expected #{expected_vocab_size}, got #{actual_vocab_size}. " \
      "Set NANOGPT_DATASET to a dataset matching the selected HF checkpoint."
    )
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

  def derive_onnx_io!(input_seed:, model:)
    logits = model.call(input_seed)
    MLX::Core.eval(logits)

    {
      "inputs" => [tensor_io_spec("tokens", input_seed)],
      "outputs" => [tensor_io_spec("logits", logits)]
    }
  end

  def tensor_io_spec(name, tensor)
    {
      "name" => name,
      "type" => dtype_name(tensor.dtype),
      "shape" => tensor.shape.map { |dim| Integer(dim) }
    }
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
      raise ArgumentError, "Unsupported NANOGPT_MODEL_DTYPE=#{value.inspect}; expected float32|float16|bfloat16"
    end
  end

  def resolve_seed(value)
    return DEFAULT_SEED if value.nil? || value.to_s.strip.empty?

    Integer(value)
  rescue ArgumentError
    raise ArgumentError, "Invalid NANOGPT_SEED=#{value.inspect}; expected integer"
  end

  def resolve_dataset_path(value)
    raw = value.nil? || value.to_s.strip.empty? ? NanoGptExample::DEFAULT_DATASET_PATH : value
    expanded = File.expand_path(raw.to_s)
    raise "Dataset not found: #{expanded}" unless File.exist?(expanded)

    expanded
  end

  def benchmark_step(timings, label)
    started_at = monotonic_now
    result = yield
    timings[label] = monotonic_now - started_at
    result
  end

  def print_timings(timings)
    total = timings.values.inject(0.0, :+)
    puts "  benchmark:"
    timings.each do |label, seconds|
      puts format("    - %<label>s: %<seconds>.2fs", label: label, seconds: seconds)
    end
    puts format("    - total: %.2fs", total)
  end

  def monotonic_now
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  def parameter_count(tree, seen = {})
    case tree
    when MLX::Core::Array
      oid = tree.object_id
      return 0 if seen.key?(oid)

      seen[oid] = true
      shape = tree.shape
      return 1 if shape.empty?

      shape.reduce(1) { |acc, dim| acc * Integer(dim) }
    when Hash
      tree.values.sum { |value| parameter_count(value, seen) }
    when Array
      tree.sum { |value| parameter_count(value, seen) }
    else
      0
    end
  end

  def dtype_name(dtype)
    if dtype.respond_to?(:name)
      dtype.name.to_s
    else
      dtype.to_s
    end
  end
end

if $PROGRAM_NAME == __FILE__
  NanoGptWebAssets.run!
end
