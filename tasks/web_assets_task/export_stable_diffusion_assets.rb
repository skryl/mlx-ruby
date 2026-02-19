# frozen_string_literal: true

require "fileutils"
require "json"
require "open3"

module StableDiffusionWebAssets
  module_function

  REPO_ROOT = File.expand_path("../..", __dir__)
  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "stable_diffusion")
  WEIGHTS_DIR = File.join(OUTPUT_DIR, "weights")
  WEIGHTS_REPO_MARKER = File.join(WEIGHTS_DIR, ".repo_id")

  MODEL_NAME_PREFIX = "stable_diffusion_pretrained_web_demo"
  DEFAULT_HF_REPO_ID = "stabilityai/sd-turbo"

  GENERATED_ASSET_FILES = %w[
    graph_ir.json
    text_encoder.graph_ir.json
    vae_decoder.graph_ir.json
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

  PYTHON_EXPORT_SCRIPT = <<~PY
    import json
    import sys
    from pathlib import Path

    repo_id = sys.argv[1]
    weights_dir = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()

    def require_modules():
      try:
        import torch
        import onnx
        import transformers
        from huggingface_hub import snapshot_download
      except Exception as exc:
        raise RuntimeError(
          "Missing Python deps for Stable Diffusion export. "
          "Install: torch diffusers transformers huggingface_hub onnx"
        ) from exc
      return torch, onnx, transformers, snapshot_download

    def ensure_transformers_compat(transformers):
      utils = transformers.utils
      if not hasattr(utils, "FLAX_WEIGHTS_NAME"):
        setattr(utils, "FLAX_WEIGHTS_NAME", "flax_model.msgpack")

    def tensor_elem_type(elem_type):
      mapping = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        9: "bool",
        10: "float16",
        11: "float64",
        12: "uint32",
        13: "uint64",
        16: "bfloat16",
      }
      return mapping.get(elem_type, f"onnx:{elem_type}")

    def shape_from_value_info(value_info):
      shape = []
      dims = value_info.type.tensor_type.shape.dim
      for dim in dims:
        if dim.HasField("dim_value"):
          shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
          shape.append(dim.dim_param)
        else:
          shape.append(-1)
      return shape

    def inspect_onnx_model(onnx, model_path):
      model = onnx.load(str(model_path))

      def serialize(value_info):
        return {
          "name": value_info.name,
          "type": tensor_elem_type(value_info.type.tensor_type.elem_type),
          "shape": shape_from_value_info(value_info),
        }

      return {
        "inputs": [serialize(entry) for entry in model.graph.input],
        "outputs": [serialize(entry) for entry in model.graph.output],
      }

    def external_data_file_name(model_path):
      sidecar = Path(str(model_path) + ".data")
      return sidecar.name if sidecar.exists() else None

    def to_json_value(value):
      if isinstance(value, (str, int, float, bool)) or value is None:
        return value
      if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
      return str(value)

    def scheduler_payload(pipeline):
      config = {}
      raw = getattr(pipeline.scheduler, "config", None)
      if raw is not None:
        try:
          for key, value in dict(raw).items():
            config[str(key)] = to_json_value(value)
        except Exception:
          pass

      config.setdefault("beta_start", 0.00085)
      config.setdefault("beta_end", 0.012)
      config.setdefault("beta_schedule", "scaled_linear")
      config.setdefault("num_train_timesteps", 1000)
      config.setdefault("prediction_type", "epsilon")
      config["num_train_timesteps"] = int(config["num_train_timesteps"])
      config["beta_start"] = float(config["beta_start"])
      config["beta_end"] = float(config["beta_end"])
      config["beta_schedule"] = str(config["beta_schedule"])
      config["prediction_type"] = str(config["prediction_type"])
      return config

    def tokenizer_payload(pipeline):
      tokenizer = pipeline.tokenizer
      model_max_length = int(getattr(tokenizer, "model_max_length", 77) or 77)
      if model_max_length <= 0 or model_max_length > 512:
        model_max_length = 77

      bos_token_id = getattr(tokenizer, "bos_token_id", None)
      eos_token_id = getattr(tokenizer, "eos_token_id", None)
      pad_token_id = getattr(tokenizer, "pad_token_id", None)

      if bos_token_id is None:
        bos_token_id = eos_token_id if eos_token_id is not None else 0
      if eos_token_id is None:
        eos_token_id = bos_token_id
      if pad_token_id is None:
        pad_token_id = eos_token_id

      return {
        "type": "clip_bpe",
        "max_length": model_max_length,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
      }

    def export_pipeline(repo_id, weights_dir, output_dir):
      torch, onnx, transformers, snapshot_download = require_modules()
      ensure_transformers_compat(transformers)
      from diffusers import StableDiffusionPipeline

      weights_dir.mkdir(parents=True, exist_ok=True)
      output_dir.mkdir(parents=True, exist_ok=True)

      snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(weights_dir),
        allow_patterns=[
          "*.json",
          "*.txt",
          "*.model",
          "*.bin",
          "*.safetensors",
          "*.pt",
          "**/*.json",
          "**/*.txt",
          "**/*.model",
          "**/*.bin",
          "**/*.safetensors",
          "**/*.pt",
          "scheduler/*",
          "tokenizer/*",
          "text_encoder/*",
          "unet/*",
          "vae/*",
        ],
        ignore_patterns=["*.onnx", "**/*.onnx"],
      )
      model_dir = Path(snapshot_path).resolve()

      kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.float32,
        "safety_checker": None,
        "feature_extractor": None,
      }
      try:
        pipeline = StableDiffusionPipeline.from_pretrained(str(model_dir), **kwargs)
      except TypeError:
        kwargs.pop("feature_extractor", None)
        pipeline = StableDiffusionPipeline.from_pretrained(str(model_dir), **kwargs)

      pipeline = pipeline.to("cpu")
      pipeline.text_encoder.eval()
      pipeline.unet.eval()
      pipeline.vae.eval()

      tokenizer_info = tokenizer_payload(pipeline)
      scheduler_info = scheduler_payload(pipeline)

      max_length = int(tokenizer_info["max_length"])
      sample_size = int(getattr(pipeline.unet.config, "sample_size", 32) or 32)
      in_channels = int(getattr(pipeline.unet.config, "in_channels", 4) or 4)
      cross_attention_dim = int(getattr(pipeline.unet.config, "cross_attention_dim", 32) or 32)
      vae_scaling_factor = float(getattr(pipeline.vae.config, "scaling_factor", 0.18215) or 0.18215)

      input_ids = torch.zeros((1, max_length), dtype=torch.int64)
      sample = torch.randn((1, in_channels, sample_size, sample_size), dtype=torch.float32)
      timestep = torch.tensor([1], dtype=torch.int64)
      encoder_hidden_states = torch.randn((1, max_length, cross_attention_dim), dtype=torch.float32)
      latent_sample = torch.randn((1, in_channels, sample_size, sample_size), dtype=torch.float32)

      class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, module):
          super().__init__()
          self.module = module

        def forward(self, input_ids):
          outputs = self.module(input_ids=input_ids, return_dict=False)
          return outputs[0]

      class UnetWrapper(torch.nn.Module):
        def __init__(self, module):
          super().__init__()
          self.module = module

        def forward(self, sample, timestep, encoder_hidden_states):
          outputs = self.module(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
          )
          return outputs[0]

      class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
          super().__init__()
          self.vae = vae

        def forward(self, latent_sample):
          outputs = self.vae.decode(latent_sample, return_dict=False)
          return outputs[0]

      text_encoder_wrapper = TextEncoderWrapper(pipeline.text_encoder)
      unet_wrapper = UnetWrapper(pipeline.unet)
      vae_decoder_wrapper = VaeDecoderWrapper(pipeline.vae)

      text_encoder_path = output_dir / "text_encoder.onnx"
      unet_path = output_dir / "unet.onnx"
      vae_decoder_path = output_dir / "vae_decoder.onnx"

      with torch.no_grad():
        torch.onnx.export(
          text_encoder_wrapper,
          (input_ids,),
          str(text_encoder_path),
          input_names=["input_ids"],
          output_names=["last_hidden_state"],
          opset_version=18,
          external_data=False,
          do_constant_folding=True,
        )
        torch.onnx.export(
          unet_wrapper,
          (sample, timestep, encoder_hidden_states),
          str(unet_path),
          input_names=["sample", "timestep", "encoder_hidden_states"],
          output_names=["out_sample"],
          opset_version=18,
          external_data=False,
          do_constant_folding=True,
        )
        torch.onnx.export(
          vae_decoder_wrapper,
          (latent_sample,),
          str(vae_decoder_path),
          input_names=["latent_sample"],
          output_names=["sample"],
          opset_version=18,
          external_data=False,
          do_constant_folding=True,
        )

      text_encoder_spec = inspect_onnx_model(onnx, text_encoder_path)
      unet_spec = inspect_onnx_model(onnx, unet_path)
      vae_decoder_spec = inspect_onnx_model(onnx, vae_decoder_path)
      is_sd_turbo = str(repo_id).strip().lower() == "stabilityai/sd-turbo"

      manifest = {
        "repo_id": repo_id,
        "weights_dir": str(model_dir),
        "pipeline": {
          "text_encoder": {
            "path": "text_encoder.onnx",
            "external_data": external_data_file_name(text_encoder_path),
            **text_encoder_spec
          },
          "unet": {
            "path": "unet.onnx",
            "external_data": external_data_file_name(unet_path),
            **unet_spec
          },
          "vae_decoder": {
            "path": "vae_decoder.onnx",
            "external_data": external_data_file_name(vae_decoder_path),
            **vae_decoder_spec
          },
        },
        "tokenizer": tokenizer_info,
        "scheduler": scheduler_info,
        "generation": {
          "latent_shape": [1, in_channels, sample_size, sample_size],
          "default_steps": 1 if is_sd_turbo else 20,
          "default_guidance_scale": 0.0 if is_sd_turbo else 7.5,
          "recommended_steps_min": 1,
          "recommended_steps_max": 4 if is_sd_turbo else 64,
          "recommended_guidance_min": 0.0,
          "recommended_guidance_max": 2.0 if is_sd_turbo else 14.0,
          "max_time": int(scheduler_info["num_train_timesteps"]) - 1,
          "vae_scaling_factor": vae_scaling_factor,
        },
      }
      return manifest

    def main():
      manifest = export_pipeline(repo_id, weights_dir, output_dir)
      print(json.dumps(manifest))

    if __name__ == "__main__":
      main()
  PY

  def run!
    repo_id = ENV.fetch("STABLE_DIFFUSION_HF_REPO", DEFAULT_HF_REPO_ID)
    python_bin = ENV.fetch("PYTHON_BIN", ENV.fetch("PYTHON", "python3"))

    FileUtils.mkdir_p(OUTPUT_DIR)
    FileUtils.mkdir_p(WEIGHTS_DIR)
    clear_weights_dir! if current_weights_repo_id != repo_id
    clear_generated_assets!

    manifest = run_python_export!(repo_id: repo_id, python_bin: python_bin)
    copy_tokenizer_assets!(manifest.fetch("weights_dir"))

    FileUtils.cp(File.join(OUTPUT_DIR, "unet.onnx"), File.join(OUTPUT_DIR, "model.onnx"))

    metadata = {
      "format" => "stable_diffusion_pretrained_web_demo_asset_v1",
      "model_name" => MODEL_NAME_PREFIX,
      "source" => "StableDiffusionPipeline.from_pretrained",
      "pipeline" => manifest.fetch("pipeline"),
      "generation" => manifest.fetch("generation"),
      "tokenizer" => manifest.fetch("tokenizer"),
      "weights" => {
        "source" => "huggingface",
        "repo_id" => repo_id,
        "trained" => true,
        "onnx_origin" => "local_export",
        "notes" => "ONNX generated locally from pretrained weights (no external ONNX artifacts)."
      }
    }

    File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
    File.binwrite(File.join(OUTPUT_DIR, "prompt.presets.json"), JSON.pretty_generate(PROMPT_PRESETS))
    File.binwrite(File.join(OUTPUT_DIR, "scheduler_config.json"), JSON.pretty_generate(manifest.fetch("scheduler")))
    File.binwrite(File.join(OUTPUT_DIR, "tokenizer_config.json"), JSON.pretty_generate(manifest.fetch("tokenizer")))
    File.binwrite(WEIGHTS_REPO_MARKER, "#{repo_id}\n")

    puts "Wrote Stable Diffusion demo assets to #{OUTPUT_DIR}"
    puts "  repo: #{repo_id}"
    puts "  - #{File.join(OUTPUT_DIR, 'text_encoder.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'unet.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'vae_decoder.onnx')}"
    puts "  - #{File.join(OUTPUT_DIR, 'model.onnx')} (compat alias for unet.onnx)"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
  end

  def run_python_export!(repo_id:, python_bin:)
    stdout, stderr, status = Open3.capture3(
      python_bin,
      "-c",
      PYTHON_EXPORT_SCRIPT,
      repo_id,
      WEIGHTS_DIR,
      OUTPUT_DIR
    )
    return parse_manifest(stdout) if status.success?

    raise(
      "Stable Diffusion local ONNX export failed.\n" \
      "Command: #{python_bin} -c <export_script> #{repo_id} #{WEIGHTS_DIR} #{OUTPUT_DIR}\n" \
      "stdout:\n#{stdout}\n" \
      "stderr:\n#{stderr}"
    )
  end

  def parse_manifest(stdout)
    line = stdout.to_s.each_line.map(&:strip).reverse.find { |entry| entry.start_with?("{") }
    raise "Stable Diffusion export did not emit JSON manifest.\nstdout:\n#{stdout}" if line.nil?

    JSON.parse(line)
  rescue JSON::ParserError => e
    raise "Failed parsing Stable Diffusion export manifest JSON: #{e.message}\nstdout:\n#{stdout}"
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
end

if $PROGRAM_NAME == __FILE__
  StableDiffusionWebAssets.run!
end
