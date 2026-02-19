# frozen_string_literal: true

require "mlx"
require "mlx/dsl"

module NanoGptExample
  REPO_ROOT = File.expand_path("../..", __dir__)
  DEFAULT_DATASET_PATH = File.expand_path("test/fixtures/karpathy.txt", REPO_ROOT)
  DEFAULT_PROMPT = "To be, or not to be"

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
      @block_size = Integer(block_size)
      @gradient_checkpointing = !!gradient_checkpointing
      self.token_embedding = MLX::NN::Embedding.new(vocab_size, n_embd)
      self.position_embedding = MLX::NN::Embedding.new(@block_size, n_embd)
      self.dropout = MLX::NN::Dropout.new(dropout.to_f)
      self.blocks = Array.new(Integer(n_layer)) do
        TransformerBlock.new(
          n_embd: n_embd,
          n_head: n_head,
          block_size: @block_size,
          dropout: dropout,
          bias: bias
        )
      end
      @checkpointed_blocks = if @gradient_checkpointing
        blocks.map { |block| MLX::NN.checkpoint(block) }
      else
        nil
      end
      self.layer_norm = MLX::NN::LayerNorm.new(n_embd, eps: 1e-5, affine: true, bias: bias)
      self.lm_head = MLX::NN::Linear.new(n_embd, vocab_size, bias: false)
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
