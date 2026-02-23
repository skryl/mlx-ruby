# frozen_string_literal: true

require "json"
require "fileutils"
require "open3"

require "mlx"
require "mlx/dsl"

module NanoGptExample
  REPO_ROOT = File.expand_path("../..", __dir__)
  DEFAULT_DATASET_PATH = File.expand_path("test/fixtures/karpathy.txt", REPO_ROOT)
  DEFAULT_PROMPT = "To be, or not to be"
  HF_REPO_ID = "sosier/nanoGPT-shakespeare-char-weights-not-tied"
  HF_REQUIRED_FILES = %w[config.json model.safetensors].freeze

  # Mirrors nanoGPT train_shakespeare_char.py defaults for model/data scale.
  REFERENCE_CONFIG = {
    dataset: "shakespeare_char",
    batch_size: 64,
    block_size: 256,
    gradient_accumulation_steps: 1,
    n_layer: 6,
    n_head: 6,
    n_embd: 384,
    dropout: 0.2,
    bias: false,
    learning_rate: 1e-3,
    max_iters: 5000,
    lr_decay_iters: 5000,
    min_lr: 1e-4,
    warmup_iters: 100,
    beta1: 0.9,
    beta2: 0.99,
    weight_decay: 0.1,
    eval_interval: 250,
    eval_iters: 200,
    log_interval: 10
  }.freeze

  Config = Struct.new(
    :vocab_size,
    :block_size,
    :n_layer,
    :n_head,
    :n_embd,
    :dropout,
    :bias,
    :layer_norm_epsilon,
    :outbedding_weight_tying,
    keyword_init: true
  ) do
    def self.from_hf_config(payload)
      new(
        vocab_size: Integer(payload.fetch("vocab_size")),
        block_size: Integer(payload.fetch("block_size")),
        n_layer: Integer(payload.fetch("n_layer")),
        n_head: Integer(payload.fetch("n_head")),
        n_embd: Integer(payload.fetch("n_embd")),
        dropout: Float(payload.fetch("dropout", 0.0)),
        bias: !!payload.fetch("bias", true),
        layer_norm_epsilon: Float(payload.fetch("layer_norm_epsilon", 1e-5)),
        outbedding_weight_tying: !!payload.fetch("outbedding_weight_tying", true)
      )
    end
  end

  class CausalSelfAttention < MLX::NN::Module
    def initialize(n_embd:, n_head:, block_size:, dropout:, bias:)
      super()

      unless (n_embd % n_head).zero?
        raise ArgumentError, "n_embd must be divisible by n_head (got n_embd=#{n_embd}, n_head=#{n_head})"
      end

      @n_head = Integer(n_head)
      @head_dim = Integer(n_embd / n_head)
      @block_size = Integer(block_size)
      @scale = @head_dim**-0.5
      self.q_proj = MLX::NN::Linear.new(n_embd, n_embd, bias: bias)
      self.k_proj = MLX::NN::Linear.new(n_embd, n_embd, bias: bias)
      self.v_proj = MLX::NN::Linear.new(n_embd, n_embd, bias: bias)
      self.out_proj = MLX::NN::Linear.new(n_embd, n_embd, bias: bias)
      self.attn_dropout = MLX::NN::Dropout.new(dropout.to_f)
      self.resid_dropout = MLX::NN::Dropout.new(dropout.to_f)
      @full_causal_mask = MLX::DSL::Masks.causal(length: @block_size, dtype: MLX::Core.float32)
    end

    def call(x)
      batch_size, sequence_length, channels = x.shape
      if sequence_length > @block_size
        raise ArgumentError, "sequence length #{sequence_length} exceeds block size #{@block_size}"
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
      return @full_causal_mask if sequence_length == @block_size

      MLX::Core.slice(@full_causal_mask, [0, 0], [sequence_length, sequence_length])
    end
  end

  class FeedForward < MLX::NN::Module
    def initialize(n_embd:, dropout:, bias:)
      super()
      self.fc = MLX::NN::Linear.new(n_embd, n_embd * 4, bias: bias)
      self.proj = MLX::NN::Linear.new(n_embd * 4, n_embd, bias: bias)
      self.dropout = MLX::NN::Dropout.new(dropout.to_f)
    end

    def call(x)
      hidden = fc.call(x)
      hidden = MLX::NN.gelu(hidden)
      hidden = proj.call(hidden)
      dropout.call(hidden)
    end
  end

  class TransformerBlock < MLX::NN::Module
    def initialize(n_embd:, n_head:, block_size:, dropout:, bias:)
      super()
      self.ln_1 = MLX::NN::LayerNorm.new(n_embd, eps: 1e-5, affine: true, bias: bias)
      self.attn = CausalSelfAttention.new(
        n_embd: n_embd,
        n_head: n_head,
        block_size: block_size,
        dropout: dropout,
        bias: bias
      )
      self.ln_2 = MLX::NN::LayerNorm.new(n_embd, eps: 1e-5, affine: true, bias: bias)
      self.mlp = FeedForward.new(n_embd: n_embd, dropout: dropout, bias: bias)
    end

    def call(x)
      x = MLX::Core.add(x, attn.call(ln_1.call(x)))
      MLX::Core.add(x, mlp.call(ln_2.call(x)))
    end
  end

  class NanoGptModel < MLX::NN::Module
    attr_reader :config

    def initialize(
      vocab_size:,
      block_size:,
      n_layer:,
      n_head:,
      n_embd:,
      dropout: 0.0,
      bias: false,
      gradient_checkpointing: false
    )
      super()
      @config = Config.new(
        vocab_size: Integer(vocab_size),
        block_size: Integer(block_size),
        n_layer: Integer(n_layer),
        n_head: Integer(n_head),
        n_embd: Integer(n_embd),
        dropout: Float(dropout),
        bias: !!bias,
        layer_norm_epsilon: 1e-5,
        outbedding_weight_tying: false
      )
      @block_size = config.block_size
      @gradient_checkpointing = !!gradient_checkpointing
      self.token_embedding = MLX::NN::Embedding.new(config.vocab_size, config.n_embd)
      self.position_embedding = MLX::NN::Embedding.new(@block_size, config.n_embd)
      self.dropout = MLX::NN::Dropout.new(config.dropout)
      self.blocks = Array.new(config.n_layer) do
        TransformerBlock.new(
          n_embd: config.n_embd,
          n_head: config.n_head,
          block_size: @block_size,
          dropout: config.dropout,
          bias: config.bias
        )
      end
      @checkpointed_blocks = if @gradient_checkpointing
        blocks.map { |block| MLX::NN.checkpoint(block) }
      else
        nil
      end
      self.layer_norm = MLX::NN::LayerNorm.new(
        config.n_embd,
        eps: config.layer_norm_epsilon,
        affine: true,
        bias: config.bias
      )
      self.lm_head = MLX::NN::Linear.new(config.n_embd, config.vocab_size, bias: false)
    end

    def call(input_ids)
      sequence_length = input_ids.shape[1]
      if sequence_length > @block_size
        raise ArgumentError, "sequence length #{sequence_length} exceeds block size #{@block_size}"
      end

      positions = MLX::Core.arange(0, sequence_length, 1, MLX::Core.int32)
      hidden = MLX::Core.add(
        token_embedding.call(input_ids),
        position_embedding.call(positions)
      )
      hidden = dropout.call(hidden)
      if @gradient_checkpointing && !@checkpointed_blocks.nil?
        @checkpointed_blocks.each { |block_fn| hidden = block_fn.call(hidden) }
      else
        blocks.each { |block| hidden = block.call(hidden) }
      end
      lm_head.call(layer_norm.call(hidden))
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

      raise "Missing HF config at #{config_path}" unless File.exist?(config_path)
      raise "Missing HF weights at #{weights_path}" unless File.exist?(weights_path)

      config = Config.from_hf_config(JSON.parse(File.binread(config_path)))
      model = new(
        vocab_size: config.vocab_size,
        block_size: config.block_size,
        n_layer: config.n_layer,
        n_head: config.n_head,
        n_embd: config.n_embd,
        dropout: config.dropout,
        bias: config.bias
      )

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
      out["layer_norm.weight"] = fetch_tensor(state, "transformer.ln_f.weight")
      if config.bias
        out["layer_norm.bias"] = fetch_tensor(state, "transformer.ln_f.bias")
      end
      out["lm_head.weight"] = if tensor_key?(state, "lm_head.weight")
        fetch_tensor(state, "lm_head.weight")
      else
        fetch_tensor(state, "transformer.wte.weight")
      end

      config.n_layer.times do |layer|
        hf_prefix = "transformer.h.#{layer}"
        model_prefix = "blocks.#{layer}"

        out["#{model_prefix}.ln_1.weight"] = fetch_tensor(state, "#{hf_prefix}.ln_1.weight")
        out["#{model_prefix}.ln_2.weight"] = fetch_tensor(state, "#{hf_prefix}.ln_2.weight")
        if config.bias
          out["#{model_prefix}.ln_1.bias"] = fetch_tensor(state, "#{hf_prefix}.ln_1.bias")
          out["#{model_prefix}.ln_2.bias"] = fetch_tensor(state, "#{hf_prefix}.ln_2.bias")
        end

        c_attn_weight = fetch_tensor(state, "#{hf_prefix}.attn.c_attn.weight")
        out["#{model_prefix}.attn.q_proj.weight"] = split_qkv_weight(c_attn_weight, 0, n_embd, n_embd)
        out["#{model_prefix}.attn.k_proj.weight"] = split_qkv_weight(c_attn_weight, n_embd, n_embd * 2, n_embd)
        out["#{model_prefix}.attn.v_proj.weight"] = split_qkv_weight(c_attn_weight, n_embd * 2, n_embd * 3, n_embd)

        if config.bias
          c_attn_bias = fetch_tensor(state, "#{hf_prefix}.attn.c_attn.bias")
          out["#{model_prefix}.attn.q_proj.bias"] = slice_1d(c_attn_bias, 0, n_embd)
          out["#{model_prefix}.attn.k_proj.bias"] = slice_1d(c_attn_bias, n_embd, n_embd * 2)
          out["#{model_prefix}.attn.v_proj.bias"] = slice_1d(c_attn_bias, n_embd * 2, n_embd * 3)
        end

        out["#{model_prefix}.attn.out_proj.weight"] = normalize_linear_weight(
          fetch_tensor(state, "#{hf_prefix}.attn.c_proj.weight"),
          out_features: n_embd,
          in_features: n_embd
        )
        if config.bias
          out["#{model_prefix}.attn.out_proj.bias"] = fetch_tensor(state, "#{hf_prefix}.attn.c_proj.bias")
        end

        out["#{model_prefix}.mlp.fc.weight"] = normalize_linear_weight(
          fetch_tensor(state, "#{hf_prefix}.mlp.c_fc.weight"),
          out_features: n_embd * 4,
          in_features: n_embd
        )
        if config.bias
          out["#{model_prefix}.mlp.fc.bias"] = fetch_tensor(state, "#{hf_prefix}.mlp.c_fc.bias")
        end
        out["#{model_prefix}.mlp.proj.weight"] = normalize_linear_weight(
          fetch_tensor(state, "#{hf_prefix}.mlp.c_proj.weight"),
          out_features: n_embd,
          in_features: n_embd * 4
        )
        if config.bias
          out["#{model_prefix}.mlp.proj.bias"] = fetch_tensor(state, "#{hf_prefix}.mlp.c_proj.bias")
        end
      end

      out
    end

    def fetch_tensor(state, key)
      candidate_tensor_keys(key).each do |candidate|
        value = state[candidate]
        if value.nil? && state.respond_to?(:key?) && state.key?(candidate.to_sym)
          value = state[candidate.to_sym]
        end
        return value if value.is_a?(MLX::Core::Array)
      end

      raise KeyError, "Missing tensor #{key} in HF nanoGPT state dict"
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

    def slice_cols(matrix, start_idx, end_idx)
      MLX::Core.slice(matrix, [0, start_idx], [matrix.shape[0], end_idx])
    end

    def slice_rows(matrix, start_idx, end_idx)
      MLX::Core.slice(matrix, [start_idx, 0], [end_idx, matrix.shape[1]])
    end

    def split_qkv_weight(c_attn_weight, start_idx, end_idx, n_embd)
      rows = Integer(c_attn_weight.shape[0])
      cols = Integer(c_attn_weight.shape[1])

      if rows == n_embd && cols == n_embd * 3
        return transpose_2d(slice_cols(c_attn_weight, start_idx, end_idx))
      end
      if rows == n_embd * 3 && cols == n_embd
        return slice_rows(c_attn_weight, start_idx, end_idx)
      end

      raise(
        "Unexpected c_attn weight shape #{c_attn_weight.shape.inspect}; " \
        "expected [#{n_embd}, #{n_embd * 3}] or [#{n_embd * 3}, #{n_embd}]"
      )
    end

    def normalize_linear_weight(weight, out_features:, in_features:)
      rows = Integer(weight.shape[0])
      cols = Integer(weight.shape[1])
      if rows == out_features && cols == in_features
        return weight
      end
      if rows == in_features && cols == out_features
        return transpose_2d(weight)
      end

      raise(
        "Unexpected linear weight shape #{weight.shape.inspect}; " \
        "expected [#{out_features}, #{in_features}] or [#{in_features}, #{out_features}]"
      )
    end

    def slice_1d(vector, start_idx, end_idx)
      MLX::Core.slice(vector, [start_idx], [end_idx])
    end

    def transpose_2d(matrix)
      MLX::Core.transpose(matrix, [1, 0])
    end
  end

  module Dataset
    module_function

    def load_text(path = DEFAULT_DATASET_PATH)
      dataset_path = File.expand_path(path.to_s)
      raise "Dataset not found: #{dataset_path}" unless File.exist?(dataset_path)

      text = File.binread(dataset_path)
      raise "Dataset is empty: #{dataset_path}" if text.empty?

      text.force_encoding(Encoding::UTF_8)
      text.encode(Encoding::UTF_8, invalid: :replace, undef: :replace, replace: "")
    end

    def build_char_tokenizer(text)
      chars = text.each_char.to_a.uniq.sort
      raise "No characters found in dataset" if chars.empty?

      char_to_id = {}
      id_to_char = {}
      chars.each_with_index do |char, index|
        char_to_id[char] = index
        id_to_char[index.to_s] = char
      end

      pad_id = Integer(char_to_id.fetch("\n", 0))
      {
        "type" => "shakespeare_char_v1",
        "vocab_size" => chars.length,
        "pad_id" => pad_id,
        "char_to_id" => char_to_id,
        "id_to_char" => id_to_char
      }
    end

    def encode(text, tokenizer)
      mapping = tokenizer.fetch("char_to_id")
      text.each_char.map do |char|
        value = mapping[char]
        raise "Tokenizer cannot encode character: #{char.inspect}" if value.nil?

        Integer(value)
      end
    end

    def decode(token_ids, tokenizer)
      mapping = tokenizer.fetch("id_to_char")
      token_ids.map { |id| mapping.fetch(Integer(id).to_s, "") }.join
    end

    def split_train_val(token_ids, train_fraction: 0.9)
      split_index = (token_ids.length * train_fraction.to_f).to_i
      split_index = 1 if split_index <= 0
      split_index = token_ids.length - 1 if split_index >= token_ids.length

      [token_ids[0...split_index], token_ids[split_index...token_ids.length]]
    end

    def sample_batch(token_ids:, batch_size:, block_size:, rng:)
      max_start = token_ids.length - block_size - 1
      if max_start <= 0
        raise "Dataset too short for block_size=#{block_size}: token_count=#{token_ids.length}"
      end

      starts = Array.new(batch_size) { rng.rand(0..max_start) }
      inputs = starts.map { |start| token_ids[start, block_size] }
      targets = starts.map { |start| token_ids[start + 1, block_size] }
      [
        MLX::Core.array(inputs, MLX::Core.int32),
        MLX::Core.array(targets, MLX::Core.int32)
      ]
    end
  end

  module Train
    module_function

    def loss_fn(model, input_ids, target_ids)
      logits = model.call(input_ids)
      batch_size, sequence_length, vocab_size = logits.shape
      flat_logits = MLX::Core.reshape(logits, [batch_size * sequence_length, vocab_size])
      flat_targets = MLX::Core.reshape(target_ids, [batch_size * sequence_length])
      MLX::NN.cross_entropy(flat_logits, flat_targets, reduction: "mean")
    end

    def learning_rate_for_step(step:, learning_rate:, min_lr:, warmup_iters:, lr_decay_iters:)
      step = Integer(step)
      learning_rate = Float(learning_rate)
      min_lr = Float(min_lr)
      warmup_iters = Integer(warmup_iters)
      lr_decay_iters = Integer(lr_decay_iters)

      if step < warmup_iters
        return learning_rate * (step + 1).to_f / warmup_iters.to_f
      end
      return min_lr if step >= lr_decay_iters

      decay_ratio = (step - warmup_iters).to_f / (lr_decay_iters - warmup_iters).to_f
      coeff = 0.5 * (1.0 + Math.cos(Math::PI * decay_ratio))
      min_lr + coeff * (learning_rate - min_lr)
    end

    def build_context(prompt_ids:, block_size:, pad_id:)
      trimmed = prompt_ids.last(block_size)
      if trimmed.length < block_size
        Array.new(block_size - trimmed.length, Integer(pad_id)) + trimmed
      else
        trimmed
      end
    end

    def sample_text(model:, tokenizer:, prompt:, block_size:, pad_id:, max_new_tokens:, temperature:, rng:)
      prompt_ids = Dataset.encode(prompt, tokenizer)
      context = build_context(prompt_ids: prompt_ids, block_size: block_size, pad_id: pad_id)
      generated_ids = []

      max_new_tokens.times do
        input = MLX::Core.array([context], MLX::Core.int32)
        logits = model.call(input)
        next_id = sample_next_id(logits, temperature: temperature, rng: rng)
        generated_ids << next_id
        context = context[1..] + [next_id]
      end

      "#{prompt}#{Dataset.decode(generated_ids, tokenizer)}"
    end

    def sample_next_id(logits, temperature:, rng:)
      index = MLX::Core.array([logits.shape[1] - 1], MLX::Core.int32)
      step_logits = MLX::Core.take(logits, index, 1)
      step_logits = MLX::Core.squeeze(step_logits, 1)

      if temperature.to_f <= 0.0
        token = MLX::Core.argmax(step_logits, -1)
        MLX::Core.eval(token)
        value = token.to_a
        value = value.first if value.is_a?(Array)
        return Integer(value)
      end

      values = step_logits.to_a
      values = values.first if values.first.is_a?(Array)
      scaled = values.map { |value| value.to_f / temperature.to_f }
      max_value = scaled.max
      weights = scaled.map { |value| Math.exp(value - max_value) }
      total = weights.sum
      threshold = rng.rand * total
      running = 0.0
      weights.each_with_index do |weight, idx|
        running += weight
        return idx if running >= threshold
      end
      weights.length - 1
    end
  end
end
