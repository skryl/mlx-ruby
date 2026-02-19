# frozen_string_literal: true

require "json"
require "fileutils"
require "open3"

require "mlx"
require "mlx/dsl"

module Gpt2Example
  REPO_ROOT = File.expand_path("..", __dir__)
  DEFAULT_TRAINING_CORPUS_PATH = File.expand_path("test/fixtures/gpt2_example.txt", REPO_ROOT)

  HF_REPO_ID = "hf-internal-testing/tiny-random-gpt2"
  HF_REQUIRED_FILES = %w[config.json model.safetensors].freeze

  Config = Struct.new(
    :vocab_size,
    :n_positions,
    :n_embd,
    :n_layer,
    :n_head,
    :dropout,
    :layer_norm_epsilon,
    keyword_init: true
  ) do
    def self.from_hf_config(payload)
      new(
        vocab_size: Integer(payload.fetch("vocab_size")),
        n_positions: Integer(payload.fetch("n_positions")),
        n_embd: Integer(payload.fetch("n_embd")),
        n_layer: Integer(payload.fetch("n_layer")),
        n_head: Integer(payload.fetch("n_head")),
        dropout: Float(payload.fetch("embd_pdrop", 0.0)),
        layer_norm_epsilon: Float(payload.fetch("layer_norm_epsilon", 1e-5))
      )
    end
  end

  class CausalSelfAttention < MLX::NN::Module
    def initialize(config)
      super()

      unless (config.n_embd % config.n_head).zero?
        raise ArgumentError, "n_embd must be divisible by n_head (#{config.n_embd} / #{config.n_head})"
      end

      @n_head = Integer(config.n_head)
      @head_dim = Integer(config.n_embd / config.n_head)
      @max_positions = Integer(config.n_positions)
      @scale = @head_dim**-0.5

      self.q_proj = MLX::NN::Linear.new(config.n_embd, config.n_embd, bias: true)
      self.k_proj = MLX::NN::Linear.new(config.n_embd, config.n_embd, bias: true)
      self.v_proj = MLX::NN::Linear.new(config.n_embd, config.n_embd, bias: true)
      self.out_proj = MLX::NN::Linear.new(config.n_embd, config.n_embd, bias: true)
      self.attn_dropout = MLX::NN::Dropout.new(config.dropout)
      self.resid_dropout = MLX::NN::Dropout.new(config.dropout)
      @full_causal_mask = MLX::DSL::Masks.causal(length: @max_positions, dtype: MLX::Core.float32)
    end

    def call(x)
      batch_size, sequence_length, channels = x.shape
      if sequence_length > @max_positions
        raise ArgumentError, "sequence length #{sequence_length} exceeds context #{@max_positions}"
      end

      queries = MLX::Core.transpose(
        MLX::Core.reshape(q_proj.call(x), [batch_size, sequence_length, @n_head, @head_dim]),
        [0, 2, 1, 3]
      )
      keys = MLX::Core.transpose(
        MLX::Core.reshape(k_proj.call(x), [batch_size, sequence_length, @n_head, @head_dim]),
        [0, 2, 1, 3]
      )
      values = MLX::Core.transpose(
        MLX::Core.reshape(v_proj.call(x), [batch_size, sequence_length, @n_head, @head_dim]),
        [0, 2, 1, 3]
      )

      mask = causal_mask(sequence_length)
      scores = MLX::Core.matmul(
        MLX::Core.multiply(queries, @scale),
        MLX::Core.transpose(keys, [0, 1, 3, 2])
      )
      scores = MLX::Core.add(scores, mask)
      probs = MLX::Core.softmax(scores.astype(MLX::Core.float32), -1).astype(scores.dtype)
      probs = attn_dropout.call(probs)
      context = MLX::Core.matmul(probs, values)
      context = MLX::Core.transpose(context, [0, 2, 1, 3])
      context = MLX::Core.reshape(context, [batch_size, sequence_length, channels])
      resid_dropout.call(out_proj.call(context))
    end

    private

    def causal_mask(sequence_length)
      return @full_causal_mask if sequence_length == @max_positions

      MLX::Core.slice(@full_causal_mask, [0, 0], [sequence_length, sequence_length])
    end
  end

  class Mlp < MLX::NN::Module
    def initialize(config)
      super()
      hidden_dims = Integer(config.n_embd * 4)
      self.fc = MLX::NN::Linear.new(config.n_embd, hidden_dims, bias: true)
      self.proj = MLX::NN::Linear.new(hidden_dims, config.n_embd, bias: true)
      self.dropout = MLX::NN::Dropout.new(config.dropout)
    end

    def call(x)
      hidden = fc.call(x)
      hidden = MLX::NN.gelu(hidden)
      hidden = proj.call(hidden)
      dropout.call(hidden)
    end
  end

  class Block < MLX::NN::Module
    def initialize(config)
      super()
      self.ln_1 = MLX::NN::LayerNorm.new(
        config.n_embd,
        eps: config.layer_norm_epsilon,
        affine: true,
        bias: true
      )
      self.attn = CausalSelfAttention.new(config)
      self.ln_2 = MLX::NN::LayerNorm.new(
        config.n_embd,
        eps: config.layer_norm_epsilon,
        affine: true,
        bias: true
      )
      self.mlp = Mlp.new(config)
    end

    def call(x)
      x = MLX::Core.add(x, attn.call(ln_1.call(x)))
      MLX::Core.add(x, mlp.call(ln_2.call(x)))
    end
  end

  class Gpt2Model < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config
      self.token_embedding = MLX::NN::Embedding.new(config.vocab_size, config.n_embd)
      self.position_embedding = MLX::NN::Embedding.new(config.n_positions, config.n_embd)
      self.dropout = MLX::NN::Dropout.new(config.dropout)
      self.blocks = Array.new(config.n_layer) { Block.new(config) }
      self.ln_f = MLX::NN::LayerNorm.new(
        config.n_embd,
        eps: config.layer_norm_epsilon,
        affine: true,
        bias: true
      )
      self.lm_head = MLX::NN::Linear.new(config.n_embd, config.vocab_size, bias: false)
    end

    def call(input_ids)
      batch_size, sequence_length = input_ids.shape
      if sequence_length > config.n_positions
        raise ArgumentError, "sequence length #{sequence_length} exceeds context #{config.n_positions}"
      end

      position_ids = MLX::Core.arange(0, sequence_length, 1, MLX::Core.int32)
      hidden = MLX::Core.add(
        token_embedding.call(input_ids),
        position_embedding.call(position_ids)
      )
      hidden = dropout.call(hidden)
      blocks.each { |block| hidden = block.call(hidden) }
      lm_head.call(ln_f.call(hidden))
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
      config_path = File.join(dir, "config.json")
      weights_path = File.join(dir, "model.safetensors")

      unless File.exist?(config_path)
        raise "Missing HF config at #{config_path}"
      end
      unless File.exist?(weights_path)
        raise "Missing HF weights at #{weights_path}"
      end

      config = Config.from_hf_config(JSON.parse(File.binread(config_path)))
      model = new(config)

      hf_state = load_hf_state_from_weights(weights_path, dtype: dtype)
      model.load_hf_state_dict!(hf_state, strict: strict)
      model
    end

    def self.download_hf_weights!(destination_dir, repo_id: HF_REPO_ID)
      destination = File.expand_path(destination_dir.to_s)
      FileUtils.mkdir_p(destination)

      HF_REQUIRED_FILES.each do |filename|
        url = "https://huggingface.co/#{repo_id}/resolve/main/#{filename}"
        target = File.join(destination, filename)
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

      npz_path = ensure_npz_from_safetensors(weights_path)
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

    def self.ensure_npz_from_safetensors(weights_path)
      npz_path = weights_path.sub(/\.safetensors\z/, ".npz")
      if File.exist?(npz_path) && File.size(npz_path).positive? && File.mtime(npz_path) >= File.mtime(weights_path)
        return npz_path
      end

      convert_safetensors_to_npz!(weights_path, npz_path)
      npz_path
    end

    def self.convert_safetensors_to_npz!(weights_path, npz_path)
      script = <<~PY
        import numpy as np
        from safetensors import safe_open
        import sys

        source = sys.argv[1]
        destination = sys.argv[2]
        tensors = {}

        with safe_open(source, framework="np", device="cpu") as reader:
          for key in reader.keys():
            tensors[key] = reader.get_tensor(key)

        np.savez(destination, **tensors)
      PY

      stdout, stderr, status = Open3.capture3("python3", "-c", script, weights_path, npz_path)
      return npz_path if status.success?

      raise(
        "Failed converting #{weights_path} to #{npz_path} via python safetensors fallback.\n" \
        "stdout:\n#{stdout}\n" \
        "stderr:\n#{stderr}"
      )
    end

    private

    def map_hf_state_dict(state)
      out = {}
      n_embd = config.n_embd

      out["token_embedding.weight"] = fetch_tensor(state, "transformer.wte.weight")
      out["position_embedding.weight"] = fetch_tensor(state, "transformer.wpe.weight")
      out["ln_f.weight"] = fetch_tensor(state, "transformer.ln_f.weight")
      out["ln_f.bias"] = fetch_tensor(state, "transformer.ln_f.bias")
      out["lm_head.weight"] = if tensor_key?(state, "lm_head.weight")
        fetch_tensor(state, "lm_head.weight")
      else
        fetch_tensor(state, "transformer.wte.weight")
      end

      config.n_layer.times do |layer|
        hf_prefix = "transformer.h.#{layer}"
        model_prefix = "blocks.#{layer}"

        out["#{model_prefix}.ln_1.weight"] = fetch_tensor(state, "#{hf_prefix}.ln_1.weight")
        out["#{model_prefix}.ln_1.bias"] = fetch_tensor(state, "#{hf_prefix}.ln_1.bias")
        out["#{model_prefix}.ln_2.weight"] = fetch_tensor(state, "#{hf_prefix}.ln_2.weight")
        out["#{model_prefix}.ln_2.bias"] = fetch_tensor(state, "#{hf_prefix}.ln_2.bias")

        c_attn_weight = fetch_tensor(state, "#{hf_prefix}.attn.c_attn.weight")
        c_attn_bias = fetch_tensor(state, "#{hf_prefix}.attn.c_attn.bias")

        out["#{model_prefix}.attn.q_proj.weight"] = transpose_2d(slice_cols(c_attn_weight, 0, n_embd))
        out["#{model_prefix}.attn.k_proj.weight"] = transpose_2d(slice_cols(c_attn_weight, n_embd, n_embd * 2))
        out["#{model_prefix}.attn.v_proj.weight"] = transpose_2d(slice_cols(c_attn_weight, n_embd * 2, n_embd * 3))
        out["#{model_prefix}.attn.q_proj.bias"] = slice_1d(c_attn_bias, 0, n_embd)
        out["#{model_prefix}.attn.k_proj.bias"] = slice_1d(c_attn_bias, n_embd, n_embd * 2)
        out["#{model_prefix}.attn.v_proj.bias"] = slice_1d(c_attn_bias, n_embd * 2, n_embd * 3)

        out["#{model_prefix}.attn.out_proj.weight"] =
          transpose_2d(fetch_tensor(state, "#{hf_prefix}.attn.c_proj.weight"))
        out["#{model_prefix}.attn.out_proj.bias"] = fetch_tensor(state, "#{hf_prefix}.attn.c_proj.bias")

        out["#{model_prefix}.mlp.fc.weight"] = transpose_2d(fetch_tensor(state, "#{hf_prefix}.mlp.c_fc.weight"))
        out["#{model_prefix}.mlp.fc.bias"] = fetch_tensor(state, "#{hf_prefix}.mlp.c_fc.bias")
        out["#{model_prefix}.mlp.proj.weight"] = transpose_2d(fetch_tensor(state, "#{hf_prefix}.mlp.c_proj.weight"))
        out["#{model_prefix}.mlp.proj.bias"] = fetch_tensor(state, "#{hf_prefix}.mlp.c_proj.bias")
      end

      out
    end

    def tensor_key?(state, key)
      candidate_tensor_keys(key).any? do |candidate|
        state.key?(candidate) || state.key?(candidate.to_sym)
      end
    end

    def candidate_tensor_keys(key)
      candidates = [key]
      if key.start_with?("transformer.")
        candidates << key.sub(/\Atransformer\./, "")
      else
        candidates << "transformer.#{key}"
      end
      candidates.uniq
    end

    def fetch_tensor(state, key)
      candidate_tensor_keys(key).each do |candidate|
        value = state[candidate]
        value = state[candidate.to_sym] if value.nil? && state.respond_to?(:key?) && state.key?(candidate.to_sym)
        return value if value.is_a?(MLX::Core::Array)
      end

      raise KeyError, "Missing tensor #{key} in HF state dict"
    end

    def slice_cols(matrix, start_col, end_col)
      rows = matrix.shape[0]
      MLX::Core.slice(matrix, [0, start_col], [rows, end_col])
    end

    def slice_1d(vector, start_idx, end_idx)
      MLX::Core.slice(vector, [start_idx], [end_idx])
    end

    def transpose_2d(matrix)
      MLX::Core.transpose(matrix, [1, 0])
    end
  end
end
