# frozen_string_literal: true

require "fileutils"
require "json"

REPO_ROOT = File.expand_path("../..", __dir__)
LIB_ROOT = File.join(REPO_ROOT, "lib")
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)

require "mlx"
require_relative "../../examples/web/stable_diffusion_example"

module StableDiffusionWebAssets
  module_function

  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "stable_diffusion")
  WEIGHTS_DIR = File.join(OUTPUT_DIR, "weights")
  WEIGHTS_REPO_MARKER = File.join(WEIGHTS_DIR, ".repo_id")

  MODEL_NAME_PREFIX = "stable_diffusion_kanji_web_demo"
  DEFAULT_HF_REPO_ID = StableDiffusionExample::HF_REPO_ID
  DEFAULT_MODEL_DTYPE = "float16"
  DEFAULT_VAE_MODEL_DTYPE = "float32"
  DEFAULT_SAMPLE_SIZE = 32

  GENERATED_ASSET_FILES = %w[
    text_encoder.onnx
    text_encoder.onnx.data
    unet.onnx
    unet.onnx.data
    vae_decoder.onnx
    vae_decoder.onnx.data
    model.onnx
    model.onnx.data
    meta.json
    prompt.presets.json
    scheduler_config.json
    tokenizer_config.json
    vocab.json
    merges.txt
    tokenizer.json
  ].freeze

  PROMPT_PRESETS = {
    "astronaut cat" => "an astronaut cat floating above the moon, cinematic lighting",
    "city at dusk" => "a futuristic city skyline at dusk, ultra detailed",
    "paper dragon" => "an origami dragon on a wooden desk, macro photo",
    "forest glow" => "a bioluminescent forest at twilight, volumetric fog"
  }.freeze

  SCHEDULER_DEFAULTS = {
    "beta_start" => 0.00085,
    "beta_end" => 0.012,
    "beta_schedule" => "scaled_linear",
    "num_train_timesteps" => 1000,
    "prediction_type" => "epsilon"
  }.freeze

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    timings = {}
    repo_id = ENV.fetch("STABLE_DIFFUSION_HF_REPO", DEFAULT_HF_REPO_ID)
    model_dtype = resolve_model_dtype(ENV.fetch("STABLE_DIFFUSION_MODEL_DTYPE", DEFAULT_MODEL_DTYPE))
    vae_model_dtype = resolve_model_dtype(ENV.fetch("STABLE_DIFFUSION_VAE_MODEL_DTYPE", DEFAULT_VAE_MODEL_DTYPE))

    benchmark_step(timings, "prepare_output_dirs") do
      FileUtils.mkdir_p(OUTPUT_DIR)
      FileUtils.mkdir_p(WEIGHTS_DIR)
      clear_weights_dir! if current_weights_repo_id != repo_id
      clear_generated_assets!
    end

    benchmark_step(timings, "fetch_pretrained_weights") do
      StableDiffusionExample.download_hf_weights!(WEIGHTS_DIR, repo_id: repo_id)
    end

    tokenizer_info = benchmark_step(timings, "load_tokenizer_config") do
      load_tokenizer_config!
    end

    scheduler_info = benchmark_step(timings, "load_scheduler_config") do
      load_scheduler_config!
    end

    benchmark_step(timings, "copy_tokenizer_assets") do
      copy_tokenizer_assets!(WEIGHTS_DIR)
    end

    text_encoder_payload = benchmark_step(timings, "load_text_encoder_weights_into_mlx") do
      StableDiffusionExample.load_text_encoder_from_hf_directory(WEIGHTS_DIR, dtype: model_dtype)
    end
    unet_payload = benchmark_step(timings, "load_unet_weights_into_mlx") do
      StableDiffusionExample.load_unet_from_hf_directory(WEIGHTS_DIR, dtype: model_dtype)
    end
    vae_payload = benchmark_step(timings, "load_autoencoder_weights_into_mlx") do
      StableDiffusionExample.load_autoencoder_from_hf_directory(WEIGHTS_DIR, dtype: vae_model_dtype)
    end

    text_encoder, text_config, text_load_note = text_encoder_payload
    unet, unet_config, unet_load_note = unet_payload
    autoencoder, vae_config, vae_load_note = vae_payload

    max_length = Integer(tokenizer_info.fetch("max_length"))
    configured_sample_size = Integer(unet_config.fetch("sample_size", 64))
    sample_size = resolve_sample_size(
      ENV.fetch("STABLE_DIFFUSION_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE.to_s),
      configured_sample_size
    )
    unet_in_channels = Integer(unet_config.fetch("in_channels", 4))

    text_hidden_size = Integer(text_config.fetch("hidden_size"))
    cross_attention = unet_config.fetch("cross_attention_dim", text_hidden_size)
    cross_attention_dim = if cross_attention.is_a?(Array)
      Integer(cross_attention.first)
    else
      Integer(cross_attention)
    end

    if cross_attention_dim != text_hidden_size
      raise(
        "UNet cross_attention_dim #{cross_attention_dim} does not match text encoder hidden_size #{text_hidden_size}"
      )
    end

    vae_latent_channels = Integer(vae_config.fetch("latent_channels", unet_in_channels))
    vae_output_channels = Integer(vae_config.fetch("out_channels", 3))
    vae_block_out_channels = Array(vae_config.fetch("block_out_channels", []))
    vae_decode_upsample_factor = if vae_block_out_channels.empty?
      8
    else
      2**[vae_block_out_channels.length - 1, 0].max
    end
    decoded_sample_size = sample_size * vae_decode_upsample_factor
    vae_scaling_factor = Float(vae_config.fetch("scaling_factor", 0.18215))

    input_ids_seed = MLX::Core.zeros([1, max_length], MLX::Core.int32)
    sample_seed = MLX::Core.zeros([1, sample_size, sample_size, unet_in_channels], model_dtype)
    timestep_seed = MLX::Core.array([1.0], model_dtype)
    conditioning_seed = MLX::Core.zeros([1, max_length, cross_attention_dim], model_dtype)
    vae_seed = MLX::Core.zeros([1, sample_size, sample_size, vae_latent_channels], vae_model_dtype)

    text_encoder_fn = lambda do |input_ids|
      text_encoder.call(input_ids).last_hidden_state
    end

    unet_fn = lambda do |sample, timestep, encoder_hidden_states|
      unet.call(sample, timestep, encoder_hidden_states)
    end

    vae_decoder_fn = lambda do |latent_sample|
      autoencoder.decode(latent_sample)
    end

    benchmark_step(timings, "export_text_encoder_binary") do
      MLX::ONNX.export_onnx(
        File.join(OUTPUT_DIR, "text_encoder.onnx"),
        text_encoder_fn,
        input_ids_seed,
        model_name: "#{MODEL_NAME_PREFIX}_text_encoder"
      )
    end

    benchmark_step(timings, "export_unet_binary") do
      MLX::ONNX.export_onnx(
        File.join(OUTPUT_DIR, "unet.onnx"),
        unet_fn,
        sample_seed,
        timestep_seed,
        conditioning_seed,
        model_name: "#{MODEL_NAME_PREFIX}_unet"
      )
    end

    benchmark_step(timings, "export_vae_decoder_binary") do
      MLX::ONNX.export_onnx(
        File.join(OUTPUT_DIR, "vae_decoder.onnx"),
        vae_decoder_fn,
        vae_seed,
        model_name: "#{MODEL_NAME_PREFIX}_vae_decoder"
      )
    end

    benchmark_step(timings, "write_unet_compat_alias") do
      FileUtils.cp(File.join(OUTPUT_DIR, "unet.onnx"), File.join(OUTPUT_DIR, "model.onnx"))
    end

    pipeline_spec = {
      "text_encoder" => {
        "path" => "text_encoder.onnx",
        "external_data" => nil,
        "inputs" => [tensor_spec("input_ids", "int32", [1, max_length])],
        "outputs" => [tensor_spec("last_hidden_state", dtype_name(model_dtype), [1, max_length, text_hidden_size])]
      },
      "unet" => {
        "path" => "unet.onnx",
        "external_data" => nil,
        "inputs" => [
          tensor_spec("sample", dtype_name(model_dtype), [1, sample_size, sample_size, unet_in_channels]),
          tensor_spec("timestep", dtype_name(model_dtype), [1]),
          tensor_spec("encoder_hidden_states", dtype_name(model_dtype), [1, max_length, cross_attention_dim])
        ],
        "outputs" => [tensor_spec("out_sample", dtype_name(model_dtype), [1, sample_size, sample_size, unet_in_channels])]
      },
      "vae_decoder" => {
        "path" => "vae_decoder.onnx",
        "external_data" => nil,
        "inputs" => [tensor_spec("latent_sample", dtype_name(vae_model_dtype), [1, sample_size, sample_size, vae_latent_channels])],
        "outputs" => [tensor_spec("sample", dtype_name(vae_model_dtype), [1, decoded_sample_size, decoded_sample_size, vae_output_channels])]
      }
    }

    generation = {
      "latent_shape" => [1, sample_size, sample_size, unet_in_channels],
      "default_steps" => 20,
      "default_guidance_scale" => 7.5,
      "recommended_steps_min" => 1,
      "recommended_steps_max" => 64,
      "recommended_guidance_min" => 0.0,
      "recommended_guidance_max" => 14.0,
      "max_time" => Integer(scheduler_info.fetch("num_train_timesteps")) - 1,
      "vae_scaling_factor" => vae_scaling_factor
    }

    text_parameter_total = parameter_count(text_encoder.parameters)
    unet_parameter_total = parameter_count(unet.parameters)
    vae_parameter_total = parameter_count(autoencoder.parameters)
    total_parameter_count = text_parameter_total + unet_parameter_total + vae_parameter_total

    metadata = {
      "format" => "stable_diffusion_web_demo_asset_v3",
      "model_name" => MODEL_NAME_PREFIX,
      "source" => "StableDiffusionExample full pipeline (HF weights from #{repo_id})",
      "pipeline" => pipeline_spec,
      "generation" => generation,
      "tokenizer" => tokenizer_info,
      "parameters" => {
        "total" => total_parameter_count,
        "text_encoder" => text_parameter_total,
        "unet" => unet_parameter_total,
        "vae" => vae_parameter_total
      },
      "weights" => {
        "source" => "huggingface",
        "repo_id" => repo_id,
        "dtype" => dtype_name(model_dtype),
        "vae_dtype" => dtype_name(vae_model_dtype),
        "trained" => true,
        "onnx_origin" => "mlx_native_export",
        "notes" => "ONNX generated locally via MLX::ONNX.export_onnx (no Python export path)."
      }
    }

    load_notes = {}
    load_notes["text_encoder"] = text_load_note unless text_load_note.nil?
    load_notes["unet"] = unet_load_note unless unet_load_note.nil?
    load_notes["vae"] = vae_load_note unless vae_load_note.nil?
    metadata["weights"]["load_notes"] = load_notes unless load_notes.empty?

    benchmark_step(timings, "write_metadata") do
      File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
      File.binwrite(File.join(OUTPUT_DIR, "prompt.presets.json"), JSON.pretty_generate(PROMPT_PRESETS))
      File.binwrite(File.join(OUTPUT_DIR, "scheduler_config.json"), JSON.pretty_generate(scheduler_info))
      File.binwrite(File.join(OUTPUT_DIR, "tokenizer_config.json"), JSON.pretty_generate(tokenizer_info))
      File.binwrite(WEIGHTS_REPO_MARKER, "#{repo_id}\n")
    end

    puts "Wrote Stable Diffusion demo assets to #{OUTPUT_DIR}"
    puts "  repo: #{repo_id}"
    puts "  dtype: text/unet=#{dtype_name(model_dtype)}, vae=#{dtype_name(vae_model_dtype)}"
    puts "  latent sample size: #{sample_size}"
    puts "  mode: full pretrained checkpoint"
    puts "  - #{File.join(OUTPUT_DIR, 'text_encoder.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'unet.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'vae_decoder.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'model.onnx')} (compat alias for unet.onnx)"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
    print_timings(timings, prefix: "  benchmark")
  end

  def tensor_spec(name, type, shape)
    {
      "name" => name,
      "type" => type,
      "shape" => shape.map { |dim| Integer(dim) }
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
      raise ArgumentError,
            "Unsupported STABLE_DIFFUSION_MODEL_DTYPE=#{value.inspect}; expected float32|float16|bfloat16"
    end
  end

  def resolve_sample_size(value, configured_sample_size)
    parsed = Integer(value)
    if parsed <= 0
      raise ArgumentError, "sample size must be positive"
    end
    unless (parsed % 8).zero?
      raise ArgumentError, "sample size must be a multiple of 8 for Stable Diffusion latent scaling"
    end
    if parsed > configured_sample_size
      raise ArgumentError,
            "sample size #{parsed} exceeds configured maximum #{configured_sample_size} from model config"
    end
    parsed
  rescue ArgumentError, TypeError => e
    raise ArgumentError,
          "Unsupported STABLE_DIFFUSION_SAMPLE_SIZE=#{value.inspect} (#{e.message})"
  end

  def dtype_name(dtype)
    if dtype.respond_to?(:name)
      dtype.name.to_s
    else
      dtype.to_s
    end
  end

  def read_json_or_default(path, fallback)
    return fallback unless File.exist?(path)

    JSON.parse(File.binread(path))
  end

  def load_scheduler_config!
    path = File.join(WEIGHTS_DIR, "scheduler", "scheduler_config.json")
    payload = read_json_or_default(path, {})

    merged = {
      "beta_start" => Float(payload.fetch("beta_start", SCHEDULER_DEFAULTS.fetch("beta_start"))),
      "beta_end" => Float(payload.fetch("beta_end", SCHEDULER_DEFAULTS.fetch("beta_end"))),
      "beta_schedule" => payload.fetch("beta_schedule", SCHEDULER_DEFAULTS.fetch("beta_schedule")).to_s,
      "num_train_timesteps" => Integer(
        payload.fetch("num_train_timesteps", SCHEDULER_DEFAULTS.fetch("num_train_timesteps"))
      ),
      "prediction_type" => payload.fetch("prediction_type", SCHEDULER_DEFAULTS.fetch("prediction_type")).to_s
    }

    merged["num_train_timesteps"] = 1000 if merged["num_train_timesteps"] <= 0
    merged
  end

  def load_tokenizer_config!
    path = File.join(WEIGHTS_DIR, "tokenizer", "tokenizer_config.json")
    payload = read_json_or_default(path, {})

    model_max_length = Integer(payload.fetch("model_max_length", 77))
    model_max_length = 77 if model_max_length <= 0 || model_max_length > 512

    eos_token_id = payload["eos_token_id"]
    eos_token_id = 49_407 if eos_token_id.nil?

    bos_token_id = payload["bos_token_id"]
    bos_token_id = eos_token_id if bos_token_id.nil?

    pad_token_id = payload["pad_token_id"]
    pad_token_id = eos_token_id if pad_token_id.nil?

    {
      "type" => "clip_bpe",
      "max_length" => model_max_length,
      "bos_token_id" => Integer(bos_token_id),
      "eos_token_id" => Integer(eos_token_id),
      "pad_token_id" => Integer(pad_token_id)
    }
  end

  def clear_generated_assets!
    GENERATED_ASSET_FILES.each do |relative_path|
      FileUtils.rm_f(File.join(OUTPUT_DIR, relative_path))
    end
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
    FileUtils.rm_f(WEIGHTS_REPO_MARKER)
  end

  def copy_tokenizer_assets!(weights_root)
    tokenizer_root = File.join(weights_root, "tokenizer")
    unless Dir.exist?(tokenizer_root)
      raise "Missing tokenizer directory in downloaded weights: #{tokenizer_root}"
    end

    copy_required_tokenizer_file!(tokenizer_root, "vocab.json")
    copy_required_tokenizer_file!(tokenizer_root, "merges.txt")
    copy_optional_tokenizer_file!(tokenizer_root, "tokenizer.json")
  end

  def copy_required_tokenizer_file!(tokenizer_root, relative_name)
    source = File.join(tokenizer_root, relative_name)
    raise "Missing tokenizer file: #{source}" unless File.exist?(source)

    FileUtils.cp(source, File.join(OUTPUT_DIR, relative_name))
  end

  def copy_optional_tokenizer_file!(tokenizer_root, relative_name)
    source = File.join(tokenizer_root, relative_name)
    return unless File.exist?(source)

    FileUtils.cp(source, File.join(OUTPUT_DIR, relative_name))
  end

  def benchmark_step(timings, label)
    started_at = monotonic_now
    result = yield
    timings[label] = monotonic_now - started_at
    result
  end

  def print_timings(timings, prefix:)
    total = timings.values.inject(0.0, :+)
    puts "#{prefix}:"
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
end

if $PROGRAM_NAME == __FILE__
  StableDiffusionWebAssets.run!
end
