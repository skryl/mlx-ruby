# frozen_string_literal: true

require "fileutils"
require "json"
require "open3"

require "mlx"

module StableDiffusionExample
  REPO_ROOT = File.expand_path("../..", __dir__)
  HF_REPO_ID = "Narsil/tiny-stable-diffusion-torch"
  HF_REQUIRED_FILES = %w[
    model_index.json
    unet/config.json
    unet/diffusion_pytorch_model.safetensors
    tokenizer/vocab.json
    tokenizer/merges.txt
    tokenizer/tokenizer_config.json
  ].freeze

  class TinyUnetModel < MLX::NN::Module
    HF_UNET_WEIGHTS = "unet/diffusion_pytorch_model.safetensors"
    HF_REQUIRED_TENSOR_KEYS = %w[
      conv_in.weight
      conv_in.bias
      down_blocks.1.resnets.0.conv1.weight
      down_blocks.1.resnets.0.conv1.bias
      mid_block.resnets.0.conv1.weight
      mid_block.resnets.0.conv1.bias
      up_blocks.1.resnets.2.conv1.weight
      up_blocks.1.resnets.2.conv1.bias
      conv_out.weight
      conv_out.bias
      down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight
      time_embedding.linear_2.weight
      time_embedding.linear_2.bias
    ].freeze

    def initialize(cross_attention_dim: 32)
      super()
      @cross_attention_dim = Integer(cross_attention_dim)

      self.conv_in = MLX::NN::Conv2d.new(4, 32, 3, padding: 1, bias: true)
      self.down = MLX::NN::Conv2d.new(32, 64, 3, padding: 1, bias: true)
      self.mid = MLX::NN::Conv2d.new(64, 64, 3, padding: 1, bias: true)
      self.up = MLX::NN::Conv2d.new(64, 32, 3, padding: 1, bias: true)
      self.conv_out = MLX::NN::Conv2d.new(32, 4, 3, padding: 1, bias: true)

      self.cond_proj = MLX::NN::Linear.new(@cross_attention_dim, 32, bias: false)
      self.time_proj = MLX::NN::Linear.new(32, 32, bias: true)
    end

    def call(latents, timestep, encoder_hidden_states)
      batch = latents.shape[0]

      cond = MLX::Core.mean(encoder_hidden_states, 1)
      cond = cond_proj.call(cond)
      cond = MLX::Core.reshape(cond, [batch, 1, 1, cond.shape[-1]])

      t = timestep_embedding(timestep, batch, latents.dtype)
      t = time_proj.call(t)
      t = MLX::Core.reshape(t, [batch, 1, 1, t.shape[-1]])

      h = conv_in.call(latents)
      h = MLX::Core.add(h, cond)
      h = MLX::Core.add(h, t)
      h = MLX::NN.silu(h)

      h = MLX::NN.silu(down.call(h))
      h = MLX::NN.silu(mid.call(h))
      h = MLX::NN.silu(up.call(h))
      conv_out.call(h)
    end

    def train(mode = true)
      mode = !!mode
      super(mode)
      modules.each do |mod|
        next if mod.equal?(self)

        mod.train(mode)
      end
      self
    end

    def eval
      train(false)
    end

    def load_hf_state_dict!(hf_state_dict, strict: true)
      mapped = map_hf_state_dict(hf_state_dict)
      load_weights(mapped, strict: strict)
      self
    end

    def self.from_hf_directory(model_dir, strict: true, dtype: nil)
      dir = File.expand_path(model_dir.to_s)
      config_path = File.join(dir, "unet", "config.json")
      weights_path = File.join(dir, HF_UNET_WEIGHTS)

      raise "Missing HF UNet config at #{config_path}" unless File.exist?(config_path)
      raise "Missing HF UNet weights at #{weights_path}" unless File.exist?(weights_path)

      config_payload = JSON.parse(File.binread(config_path))
      cross_attention_dim = Integer(config_payload.fetch("cross_attention_dim", 32))
      model = new(cross_attention_dim: cross_attention_dim)

      hf_state = load_hf_state_from_weights(weights_path, dtype: dtype)
      model.load_hf_state_dict!(hf_state, strict: strict)
      model
    end

    def self.download_hf_weights!(destination_dir, repo_id: HF_REPO_ID)
      destination = File.expand_path(destination_dir.to_s)
      FileUtils.mkdir_p(destination)

      HF_REQUIRED_FILES.each do |relative_path|
        url = "https://huggingface.co/#{repo_id}/resolve/main/#{relative_path}"
        target = File.join(destination, relative_path)
        FileUtils.mkdir_p(File.dirname(target))
        next if File.exist?(target) && File.size(target).positive?

        system("curl", "-fL", "--retry", "3", "--retry-delay", "2", "-o", target, url)
        next if $?.success?

        raise "Failed downloading #{url}"
      end

      destination
    end

    def self.load_hf_state_from_weights(weights_path, dtype: nil)
      raw_state = MLX::Core.load(weights_path)
      normalize_hf_state(raw_state, dtype: dtype)
    rescue StandardError => e
      raise unless safetensors_native_unavailable?(e) && weights_path.end_with?(".safetensors")

      npz_path = ensure_npz_from_safetensors(weights_path, keys: HF_REQUIRED_TENSOR_KEYS)
      raw_state = MLX::Core.load(npz_path)
      normalize_hf_state(raw_state, dtype: dtype)
    end

    def self.normalize_hf_state(raw_state, dtype: nil)
      raw_state.to_a.each_with_object({}) do |(key, value), out|
        tensor = value
        tensor = tensor.astype(dtype) unless dtype.nil?
        out[key.to_s] = tensor
      end
    end

    def self.safetensors_native_unavailable?(error)
      error.message.include?("MLX_BUILD_SAFETENSORS=ON")
    end

    def self.ensure_npz_from_safetensors(weights_path, keys: nil)
      suffix = keys.nil? ? "" : ".mlx_subset"
      npz_path = weights_path.sub(/\.safetensors\z/, "#{suffix}.npz")
      if File.exist?(npz_path) && File.size(npz_path).positive? && File.mtime(npz_path) >= File.mtime(weights_path)
        return npz_path
      end

      convert_safetensors_to_npz!(weights_path, npz_path, keys: keys)
      npz_path
    end

    def self.convert_safetensors_to_npz!(weights_path, npz_path, keys: nil)
      script = <<~PY
        import json
        import numpy as np
        from safetensors import safe_open
        import sys

        source = sys.argv[1]
        destination = sys.argv[2]
        selected = None
        if len(sys.argv) > 3:
          selected = json.loads(sys.argv[3])
        tensors = {}

        with safe_open(source, framework="np", device="cpu") as reader:
          keys = selected if selected is not None else list(reader.keys())
          for key in keys:
            tensors[key] = reader.get_tensor(key)

        np.savez(destination, **tensors)
      PY

      argv = ["python3", "-c", script, weights_path, npz_path]
      argv << keys.to_json unless keys.nil?
      stdout, stderr, status = Open3.capture3(*argv)
      return npz_path if status.success?

      raise(
        "Failed converting #{weights_path} to #{npz_path} via python safetensors fallback.\n" \
        "stdout:\n#{stdout}\n" \
        "stderr:\n#{stderr}"
      )
    end

    private

    def timestep_embedding(timestep, batch, dtype)
      t = timestep
      if t.shape.empty?
        scalar = t.to_a
        scalar = scalar.first if scalar.is_a?(Array)
        t = MLX::Core.array(Array.new(batch, scalar.to_f), dtype)
      elsif t.shape.length == 1
        t = MLX::Core.broadcast_to(t, [batch]) if t.shape[0] != batch
      else
        t = MLX::Core.reshape(t, [batch])
      end

      t = MLX::Core.reshape(t, [batch, 1])
      MLX::Core.broadcast_to(t, [batch, 32])
    end

    def map_hf_state_dict(state)
      out = {}

      out["conv_in.weight"] = pytorch_conv_to_mlx(fetch_tensor(state, "conv_in.weight"))
      out["conv_in.bias"] = fetch_tensor(state, "conv_in.bias")

      out["down.weight"] = pytorch_conv_to_mlx(fetch_tensor(state, "down_blocks.1.resnets.0.conv1.weight"))
      out["down.bias"] = fetch_tensor(state, "down_blocks.1.resnets.0.conv1.bias")

      out["mid.weight"] = pytorch_conv_to_mlx(fetch_tensor(state, "mid_block.resnets.0.conv1.weight"))
      out["mid.bias"] = fetch_tensor(state, "mid_block.resnets.0.conv1.bias")

      out["up.weight"] = pytorch_conv_to_mlx(fetch_tensor(state, "up_blocks.1.resnets.2.conv1.weight"))
      out["up.bias"] = fetch_tensor(state, "up_blocks.1.resnets.2.conv1.bias")

      out["conv_out.weight"] = pytorch_conv_to_mlx(fetch_tensor(state, "conv_out.weight"))
      out["conv_out.bias"] = fetch_tensor(state, "conv_out.bias")

      cond_proj = fetch_tensor(state, "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight")
      in_dims = [@cross_attention_dim, cond_proj.shape[1]].min
      out["cond_proj.weight"] = slice_2d(cond_proj, 0, 32, 0, in_dims)

      time_weight = fetch_tensor(state, "time_embedding.linear_2.weight")
      time_bias = fetch_tensor(state, "time_embedding.linear_2.bias")
      out["time_proj.weight"] = slice_2d(time_weight, 0, 32, 0, 32)
      out["time_proj.bias"] = slice_1d(time_bias, 0, 32)

      out
    end

    def fetch_tensor(state, key)
      keys = [key, "unet.#{key}"]
      keys.each do |candidate|
        value = state[candidate]
        value = state[candidate.to_sym] if value.nil? && state.respond_to?(:key?) && state.key?(candidate.to_sym)
        return value if value.is_a?(MLX::Core::Array)
      end

      raise KeyError, "Missing tensor #{key} in HF UNet state dict"
    end

    def pytorch_conv_to_mlx(weight)
      MLX::Core.transpose(weight, [0, 2, 3, 1])
    end

    def slice_1d(vector, start_idx, end_idx)
      MLX::Core.slice(vector, [start_idx], [end_idx])
    end

    def slice_2d(matrix, row_start, row_end, col_start, col_end)
      MLX::Core.slice(matrix, [row_start, col_start], [row_end, col_end])
    end
  end
end
