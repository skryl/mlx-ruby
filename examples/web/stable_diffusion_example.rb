# frozen_string_literal: true

require "fileutils"
require "json"
require "open3"

require "mlx"
require "mlx/dsl"

module StableDiffusionExample
  REPO_ROOT = File.expand_path("../..", __dir__)
  HF_REPO_ID = "Ksgk-fy/stable-diffusion-v1-5-smaller-unet-kanji_99"

  HF_REQUIRED_FILES = %w[
    model_index.json
    text_encoder/config.json
    text_encoder/model.safetensors
    unet/config.json
    unet/diffusion_pytorch_model.safetensors
    vae/config.json
    vae/diffusion_pytorch_model.safetensors
    tokenizer/vocab.json
    tokenizer/merges.txt
    tokenizer/tokenizer_config.json
    scheduler/scheduler_config.json
  ].freeze

  class AutoencoderConfig
    include MLX::DSL::ConfigSchema

    field :in_channels, Integer, default: 3
    field :out_channels, Integer, default: 3
    field :latent_channels_out, Integer, default: 8
    field :latent_channels_in, Integer, default: 4
    field :block_out_channels, Array, default: [128, 256, 512, 512]
    field :layers_per_block, Integer, default: 2
    field :norm_num_groups, Integer, default: 32
    field :scaling_factor, [Integer, Float], default: 0.18215
  end

  class CLIPTextModelConfig
    include MLX::DSL::ConfigSchema

    field :num_layers, Integer, default: 23
    field :model_dims, Integer, default: 768
    field :num_heads, Integer, default: 12
    field :max_length, Integer, default: 77
    field :vocab_size, Integer, default: 49_408
    field :projection_dim, [Integer, NilClass], default: nil
    field :hidden_act, String, default: "quick_gelu"
  end

  class UNetConfig
    include MLX::DSL::ConfigSchema

    field :in_channels, Integer, default: 4
    field :out_channels, Integer, default: 4
    field :conv_in_kernel, Integer, default: 3
    field :conv_out_kernel, Integer, default: 3
    field :block_out_channels, Array, default: [320, 640, 1280, 1280]
    field :layers_per_block, Array, default: [2, 2, 2, 2]
    field :mid_block_layers, Integer, default: 2
    field :transformer_layers_per_block, Array, default: [1, 1, 1, 1]
    field :num_attention_heads, Array, default: [8, 8, 8, 8]
    field :cross_attention_dim, Array, default: [768, 768, 768, 768]
    field :norm_num_groups, Integer, default: 32
    field :down_block_types, Array, default: ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
    field :up_block_types, Array, default: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
    field :addition_embed_type, [String, NilClass], default: nil
    field :addition_time_embed_dim, [Integer, NilClass], default: nil
    field :projection_class_embeddings_input_dim, [Integer, NilClass], default: nil
  end

  class CLIPOutput
    attr_reader :pooled_output, :last_hidden_state, :hidden_states

    def initialize(pooled_output:, last_hidden_state:, hidden_states:)
      @pooled_output = pooled_output
      @last_hidden_state = last_hidden_state
      @hidden_states = hidden_states
    end
  end

  class CLIPEncoderLayer < MLX::NN::Module
    def initialize(model_dims, num_heads, activation)
      super()
      @activation = activation

      self.layer_norm1 = MLX::NN::LayerNorm.new(model_dims)
      self.layer_norm2 = MLX::NN::LayerNorm.new(model_dims)

      self.attention = MLX::NN::MultiHeadAttention.new(model_dims, num_heads, bias: true)
      self.attention.query_proj.bias = MLX::Core.zeros([model_dims])
      self.attention.key_proj.bias = MLX::Core.zeros([model_dims])
      self.attention.value_proj.bias = MLX::Core.zeros([model_dims])
      self.attention.out_proj.bias = MLX::Core.zeros([model_dims])

      self.linear1 = MLX::NN::Linear.new(model_dims, 4 * model_dims)
      self.linear2 = MLX::NN::Linear.new(4 * model_dims, model_dims)
    end

    def call(x, attn_mask = nil)
      y = layer_norm1.call(x)
      y = attention.call(y, y, y, attn_mask)
      x = MLX::Core.add(y, x)

      y = layer_norm2.call(x)
      y = linear1.call(y)
      y = activate(y)
      y = linear2.call(y)
      MLX::Core.add(y, x)
    end

    private

    def activate(x)
      if @activation == "gelu"
        MLX::NN.gelu(x)
      else
        MLX::Core.multiply(x, MLX::Core.sigmoid(MLX::Core.multiply(1.702, x)))
      end
    end
  end

  class CLIPTextModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config

      self.token_embedding = MLX::NN::Embedding.new(config.vocab_size, config.model_dims)
      self.position_embedding = MLX::NN::Embedding.new(config.max_length, config.model_dims)
      self.layers = Array.new(config.num_layers) do
        CLIPEncoderLayer.new(config.model_dims, config.num_heads, config.hidden_act)
      end
      self.final_layer_norm = MLX::NN::LayerNorm.new(config.model_dims)
      self.text_projection = if config.projection_dim.nil?
        nil
      else
        MLX::NN::Linear.new(config.model_dims, config.projection_dim, bias: false)
      end
    end

    def call(x)
      _b, n = x.shape
      eos_tokens = MLX::Core.argmax(x, -1)

      h = token_embedding.call(x)
      pos = MLX::Core.slice(position_embedding.weight, [0, 0], [n, position_embedding.weight.shape[1]])
      h = MLX::Core.add(h, pos)

      mask = causal_mask(n, h.dtype)
      hidden_states = []
      layers.each do |layer|
        h = layer.call(h, mask)
        hidden_states << h
      end

      h = final_layer_norm.call(h)
      pooled = select_eos(h, eos_tokens)
      pooled = text_projection.call(pooled) unless text_projection.nil?

      CLIPOutput.new(
        pooled_output: pooled,
        last_hidden_state: h,
        hidden_states: hidden_states
      )
    end

    private

    def causal_mask(n, dtype)
      idx = MLX::Core.arange(0, n, 1)
      lhs = MLX::Core.expand_dims(idx, 1)
      rhs = MLX::Core.expand_dims(idx, 0)
      mask = MLX::Core.less(lhs, rhs).astype(dtype)
      scale = (dtype == MLX::Core.float16 ? -6e4 : -1e9)
      MLX::Core.multiply(mask, MLX::Core.array(scale, dtype))
    end

    def select_eos(x, eos_tokens)
      b = x.shape[0]
      d = x.shape[2]
      idx = eos_tokens.astype(MLX::Core.int32)
      idx = MLX::Core.maximum(idx, 0)
      idx = MLX::Core.minimum(idx, x.shape[1] - 1)
      idx = MLX::Core.reshape(idx, [b, 1, 1])
      idx = MLX::Core.broadcast_to(idx, [b, 1, d])
      selected = MLX::Core.take_along_axis(x, idx, 1)
      MLX::Core.squeeze(selected, 1)
    end
  end

  def self.upsample_nearest(x, scale: 2)
    factor = scale.to_f
    MLX::NN.upsample_nearest(x, [factor, factor])
  end

  class TimestepEmbedding < MLX::NN::Module
    def initialize(in_channels, time_embed_dim)
      super()

      self.linear_1 = MLX::NN::Linear.new(in_channels, time_embed_dim)
      self.linear_2 = MLX::NN::Linear.new(time_embed_dim, time_embed_dim)
    end

    def call(x)
      x = linear_1.call(x)
      x = MLX::NN.silu(x)
      linear_2.call(x)
    end
  end

  class TransformerBlock < MLX::NN::Module
    def initialize(model_dims, num_heads, hidden_dims: nil, memory_dims: nil)
      super()

      self.norm1 = MLX::NN::LayerNorm.new(model_dims)
      self.attn1 = MLX::NN::MultiHeadAttention.new(model_dims, num_heads)
      self.attn1.out_proj.bias = MLX::Core.zeros([model_dims])

      memory_dims ||= model_dims
      self.norm2 = MLX::NN::LayerNorm.new(model_dims)
      self.attn2 = MLX::NN::MultiHeadAttention.new(model_dims, num_heads, key_input_dims: memory_dims)
      self.attn2.out_proj.bias = MLX::Core.zeros([model_dims])

      hidden_dims ||= (4 * model_dims)
      self.norm3 = MLX::NN::LayerNorm.new(model_dims)
      self.linear1 = MLX::NN::Linear.new(model_dims, hidden_dims)
      self.linear2 = MLX::NN::Linear.new(model_dims, hidden_dims)
      self.linear3 = MLX::NN::Linear.new(hidden_dims, model_dims)
    end

    def call(x, memory, attn_mask = nil, memory_mask = nil)
      y = norm1.call(x)
      y = attn1.call(y, y, y, attn_mask)
      x = MLX::Core.add(x, y)

      y = norm2.call(x)
      y = attn2.call(y, memory, memory, memory_mask)
      x = MLX::Core.add(x, y)

      y = norm3.call(x)
      y_a = linear1.call(y)
      y_b = linear2.call(y)
      y = MLX::Core.multiply(y_a, MLX::NN.gelu(y_b))
      y = linear3.call(y)
      MLX::Core.add(x, y)
    end
  end

  class Transformer2D < MLX::NN::Module
    def initialize(in_channels:, model_dims:, encoder_dims:, num_heads:, num_layers: 1, norm_num_groups: 32)
      super()

      self.norm = MLX::NN::GroupNorm.new(norm_num_groups, in_channels, pytorch_compatible: true)
      self.proj_in = MLX::NN::Linear.new(in_channels, model_dims)
      self.transformer_blocks = Array.new(num_layers) do
        TransformerBlock.new(model_dims, num_heads, memory_dims: encoder_dims)
      end
      self.proj_out = MLX::NN::Linear.new(model_dims, in_channels)
    end

    def call(x, encoder_x, attn_mask = nil, encoder_attn_mask = nil)
      input_x = x
      batch, height, width, channels = x.shape

      x = norm.call(x)
      x = MLX::Core.reshape(x, [batch, height * width, channels])
      x = proj_in.call(x)

      transformer_blocks.each do |block|
        x = block.call(x, encoder_x, attn_mask, encoder_attn_mask)
      end

      x = proj_out.call(x)
      x = MLX::Core.reshape(x, [batch, height, width, channels])
      MLX::Core.add(x, input_x)
    end
  end

  class ResnetBlock2D < MLX::NN::Module
    def initialize(in_channels:, out_channels: nil, groups: 32, temb_channels: nil)
      super()

      out_channels ||= in_channels

      self.norm1 = MLX::NN::GroupNorm.new(groups, in_channels, pytorch_compatible: true)
      self.conv1 = MLX::NN::Conv2d.new(in_channels, out_channels, 3, stride: 1, padding: 1)
      self.time_emb_proj = MLX::NN::Linear.new(temb_channels, out_channels) unless temb_channels.nil?
      self.norm2 = MLX::NN::GroupNorm.new(groups, out_channels, pytorch_compatible: true)
      self.conv2 = MLX::NN::Conv2d.new(out_channels, out_channels, 3, stride: 1, padding: 1)

      self.conv_shortcut = MLX::NN::Linear.new(in_channels, out_channels) if in_channels != out_channels
    end

    def call(x, temb = nil)
      unless temb.nil? || !respond_to?(:time_emb_proj)
        temb = time_emb_proj.call(MLX::NN.silu(temb))
      end

      y = norm1.call(x)
      y = MLX::NN.silu(y)
      y = conv1.call(y)

      unless temb.nil?
        temb = MLX::Core.reshape(temb, [temb.shape[0], 1, 1, temb.shape[1]])
        y = MLX::Core.add(y, temb)
      end

      y = norm2.call(y)
      y = MLX::NN.silu(y)
      y = conv2.call(y)

      shortcut = if respond_to?(:conv_shortcut)
        conv_shortcut.call(x)
      else
        x
      end
      MLX::Core.add(y, shortcut)
    end
  end

  class UNetBlock2D < MLX::NN::Module
    def initialize(
      in_channels:,
      out_channels:,
      temb_channels:,
      prev_out_channels: nil,
      num_layers: 1,
      transformer_layers_per_block: 1,
      num_attention_heads: 8,
      cross_attention_dim: 1280,
      resnet_groups: 32,
      add_downsample: true,
      add_upsample: true,
      add_cross_attention: true
    )
      super()

      if prev_out_channels.nil?
        in_channels_list = [in_channels] + Array.new(num_layers - 1, out_channels)
      else
        in_channels_list = [prev_out_channels] + Array.new(num_layers - 1, out_channels)
        res_channels_list = Array.new(num_layers - 1, out_channels) + [in_channels]
        in_channels_list = in_channels_list.each_with_index.map { |value, idx| value + res_channels_list[idx] }
      end

      self.resnets = in_channels_list.map do |ic|
        ResnetBlock2D.new(
          in_channels: ic,
          out_channels: out_channels,
          temb_channels: temb_channels,
          groups: resnet_groups
        )
      end

      if add_cross_attention
        self.attentions = Array.new(num_layers) do
          Transformer2D.new(
            in_channels: out_channels,
            model_dims: out_channels,
            num_heads: num_attention_heads,
            num_layers: transformer_layers_per_block,
            encoder_dims: cross_attention_dim
          )
        end
      end

      if add_downsample
        self.downsample = MLX::NN::Conv2d.new(out_channels, out_channels, 3, stride: 2, padding: 1)
      end

      if add_upsample
        self.upsample = MLX::NN::Conv2d.new(out_channels, out_channels, 3, stride: 1, padding: 1)
      end
    end

    def call(
      x,
      encoder_x: nil,
      temb: nil,
      attn_mask: nil,
      encoder_attn_mask: nil,
      residual_hidden_states: nil
    )
      output_states = []

      resnets.each_with_index do |resnet, index|
        unless residual_hidden_states.nil? || residual_hidden_states.empty?
          x = MLX::Core.concatenate([x, residual_hidden_states.pop], -1)
        end

        x = resnet.call(x, temb)

        if respond_to?(:attentions)
          x = attentions[index].call(x, encoder_x, attn_mask, encoder_attn_mask)
        end

        output_states << x
      end

      if respond_to?(:downsample)
        x = downsample.call(x)
        output_states << x
      end

      if respond_to?(:upsample)
        x = upsample.call(StableDiffusionExample.upsample_nearest(x))
        output_states << x
      end

      [x, output_states]
    end
  end

  class UNetModel < MLX::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      @config = config

      self.conv_in = MLX::NN::Conv2d.new(
        config.in_channels,
        config.block_out_channels[0],
        config.conv_in_kernel,
        padding: (config.conv_in_kernel - 1) / 2
      )

      self.timesteps = MLX::NN::SinusoidalPositionalEncoding.new(
        config.block_out_channels[0],
        max_freq: 1,
        min_freq: Math.exp(-Math.log(10_000) + ((2 * Math.log(10_000)) / config.block_out_channels[0])),
        scale: 1.0,
        cos_first: true,
        full_turns: false
      )
      self.time_embedding = TimestepEmbedding.new(config.block_out_channels[0], config.block_out_channels[0] * 4)

      if config.addition_embed_type == "text_time"
        self.add_time_proj = MLX::NN::SinusoidalPositionalEncoding.new(
          config.addition_time_embed_dim,
          max_freq: 1,
          min_freq: Math.exp(-Math.log(10_000) + ((2 * Math.log(10_000)) / config.addition_time_embed_dim)),
          scale: 1.0,
          cos_first: true,
          full_turns: false
        )
        self.add_embedding = TimestepEmbedding.new(
          config.projection_class_embeddings_input_dim,
          config.block_out_channels[0] * 4
        )
      end

      block_channels = [config.block_out_channels[0]] + config.block_out_channels
      self.down_blocks = block_channels.each_cons(2).each_with_index.map do |(in_ch, out_ch), index|
        UNetBlock2D.new(
          in_channels: in_ch,
          out_channels: out_ch,
          temb_channels: config.block_out_channels[0] * 4,
          num_layers: config.layers_per_block[index],
          transformer_layers_per_block: config.transformer_layers_per_block[index],
          num_attention_heads: config.num_attention_heads[index],
          cross_attention_dim: config.cross_attention_dim[index],
          resnet_groups: config.norm_num_groups,
          add_downsample: index < config.block_out_channels.length - 1,
          add_upsample: false,
          add_cross_attention: config.down_block_types[index].include?("CrossAttn")
        )
      end

      self.mid_blocks = [
        ResnetBlock2D.new(
          in_channels: config.block_out_channels[-1],
          out_channels: config.block_out_channels[-1],
          temb_channels: config.block_out_channels[0] * 4,
          groups: config.norm_num_groups
        ),
        Transformer2D.new(
          in_channels: config.block_out_channels[-1],
          model_dims: config.block_out_channels[-1],
          num_heads: config.num_attention_heads[-1],
          num_layers: config.transformer_layers_per_block[-1],
          encoder_dims: config.cross_attention_dim[-1]
        ),
        ResnetBlock2D.new(
          in_channels: config.block_out_channels[-1],
          out_channels: config.block_out_channels[-1],
          temb_channels: config.block_out_channels[0] * 4,
          groups: config.norm_num_groups
        )
      ]

      up_channels = [config.block_out_channels[0]] + config.block_out_channels + [config.block_out_channels[-1]]
      blocks = []
      up_channels.each_cons(3).each_with_index do |(in_ch, out_ch, prev_out_ch), index|
        blocks << [index, in_ch, out_ch, prev_out_ch]
      end

      self.up_blocks = blocks.reverse.map do |index, in_ch, out_ch, prev_out_ch|
        UNetBlock2D.new(
          in_channels: in_ch,
          out_channels: out_ch,
          temb_channels: config.block_out_channels[0] * 4,
          prev_out_channels: prev_out_ch,
          num_layers: config.layers_per_block[index] + 1,
          transformer_layers_per_block: config.transformer_layers_per_block[index],
          num_attention_heads: config.num_attention_heads[index],
          cross_attention_dim: config.cross_attention_dim[index],
          resnet_groups: config.norm_num_groups,
          add_downsample: false,
          add_upsample: index.positive?,
          add_cross_attention: config.up_block_types[index].include?("CrossAttn")
        )
      end

      self.conv_norm_out = MLX::NN::GroupNorm.new(
        config.norm_num_groups,
        config.block_out_channels[0],
        pytorch_compatible: true
      )
      self.conv_out = MLX::NN::Conv2d.new(
        config.block_out_channels[0],
        config.out_channels,
        config.conv_out_kernel,
        padding: (config.conv_out_kernel - 1) / 2
      )
    end

    def call(x, timestep, encoder_x, attn_mask = nil, encoder_attn_mask = nil, text_time = nil)
      temb = timesteps.call(timestep)
      temb = temb.astype(x.dtype)
      temb = time_embedding.call(temb)

      unless text_time.nil? || !respond_to?(:add_time_proj)
        text_emb, time_ids = text_time
        emb = add_time_proj.call(time_ids)
        emb = MLX::Core.reshape(emb, [emb.shape[0], emb.shape[1] * emb.shape[2]])
        emb = emb.astype(x.dtype)
        emb = MLX::Core.concatenate([text_emb, emb], -1)
        emb = add_embedding.call(emb)
        temb = MLX::Core.add(temb, emb)
      end

      x = conv_in.call(x)

      residuals = [x]
      down_blocks.each do |block|
        x, block_residuals = block.call(
          x,
          encoder_x: encoder_x,
          temb: temb,
          attn_mask: attn_mask,
          encoder_attn_mask: encoder_attn_mask
        )
        residuals.concat(block_residuals)
      end

      x = mid_blocks[0].call(x, temb)
      x = mid_blocks[1].call(x, encoder_x, attn_mask, encoder_attn_mask)
      x = mid_blocks[2].call(x, temb)

      up_blocks.each do |block|
        x, _ = block.call(
          x,
          encoder_x: encoder_x,
          temb: temb,
          attn_mask: attn_mask,
          encoder_attn_mask: encoder_attn_mask,
          residual_hidden_states: residuals
        )
      end

      x = conv_norm_out.call(x)
      x = MLX::NN.silu(x)
      conv_out.call(x)
    end
  end

  class Attention < MLX::NN::Module
    def initialize(dims, norm_groups: 32)
      super()

      self.group_norm = MLX::NN::GroupNorm.new(norm_groups, dims, pytorch_compatible: true)
      self.query_proj = MLX::NN::Linear.new(dims, dims)
      self.key_proj = MLX::NN::Linear.new(dims, dims)
      self.value_proj = MLX::NN::Linear.new(dims, dims)
      self.out_proj = MLX::NN::Linear.new(dims, dims)
    end

    def call(x)
      batch, height, width, channels = x.shape

      y = group_norm.call(x)

      queries = query_proj.call(y)
      queries = MLX::Core.reshape(queries, [batch, height * width, channels])
      keys = key_proj.call(y)
      keys = MLX::Core.reshape(keys, [batch, height * width, channels])
      values = value_proj.call(y)
      values = MLX::Core.reshape(values, [batch, height * width, channels])

      scale = 1.0 / Math.sqrt(channels)
      scores = MLX::Core.multiply(queries, scale)
      scores = MLX::Core.matmul(scores, MLX::Core.transpose(keys, [0, 2, 1]))
      attn = MLX::Core.softmax(scores, -1)
      y = MLX::Core.matmul(attn, values)
      y = MLX::Core.reshape(y, [batch, height, width, channels])

      y = out_proj.call(y)
      MLX::Core.add(x, y)
    end
  end

  class EncoderDecoderBlock2D < MLX::NN::Module
    def initialize(
      in_channels:,
      out_channels:,
      num_layers: 1,
      resnet_groups: 32,
      add_downsample: true,
      add_upsample: true
    )
      super()

      self.resnets = Array.new(num_layers) do |index|
        ResnetBlock2D.new(
          in_channels: (index.zero? ? in_channels : out_channels),
          out_channels: out_channels,
          groups: resnet_groups
        )
      end

      if add_downsample
        self.downsample = MLX::NN::Conv2d.new(out_channels, out_channels, 3, stride: 2, padding: 0)
      end

      if add_upsample
        self.upsample = MLX::NN::Conv2d.new(out_channels, out_channels, 3, stride: 1, padding: 1)
      end
    end

    def call(x)
      resnets.each { |resnet| x = resnet.call(x) }

      if respond_to?(:downsample)
        x = MLX::Core.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])
        x = downsample.call(x)
      end

      if respond_to?(:upsample)
        x = upsample.call(StableDiffusionExample.upsample_nearest(x))
      end

      x
    end
  end

  class Encoder < MLX::NN::Module
    def initialize(
      in_channels:,
      out_channels:,
      block_out_channels: [64],
      layers_per_block: 2,
      resnet_groups: 32
    )
      super()

      self.conv_in = MLX::NN::Conv2d.new(in_channels, block_out_channels[0], 3, stride: 1, padding: 1)

      channels = [block_out_channels[0]] + block_out_channels
      self.down_blocks = channels.each_cons(2).each_with_index.map do |(in_ch, out_ch), index|
        EncoderDecoderBlock2D.new(
          in_channels: in_ch,
          out_channels: out_ch,
          num_layers: layers_per_block,
          resnet_groups: resnet_groups,
          add_downsample: index < block_out_channels.length - 1,
          add_upsample: false
        )
      end

      self.mid_blocks = [
        ResnetBlock2D.new(in_channels: block_out_channels[-1], out_channels: block_out_channels[-1], groups: resnet_groups),
        Attention.new(block_out_channels[-1], norm_groups: resnet_groups),
        ResnetBlock2D.new(in_channels: block_out_channels[-1], out_channels: block_out_channels[-1], groups: resnet_groups)
      ]

      self.conv_norm_out = MLX::NN::GroupNorm.new(resnet_groups, block_out_channels[-1], pytorch_compatible: true)
      self.conv_out = MLX::NN::Conv2d.new(block_out_channels[-1], out_channels, 3, padding: 1)
    end

    def call(x)
      x = conv_in.call(x)
      down_blocks.each { |layer| x = layer.call(x) }

      x = mid_blocks[0].call(x)
      x = mid_blocks[1].call(x)
      x = mid_blocks[2].call(x)

      x = conv_norm_out.call(x)
      x = MLX::NN.silu(x)
      conv_out.call(x)
    end
  end

  class Decoder < MLX::NN::Module
    def initialize(
      in_channels:,
      out_channels:,
      block_out_channels: [64],
      layers_per_block: 2,
      resnet_groups: 32
    )
      super()

      self.conv_in = MLX::NN::Conv2d.new(in_channels, block_out_channels[-1], 3, stride: 1, padding: 1)

      self.mid_blocks = [
        ResnetBlock2D.new(in_channels: block_out_channels[-1], out_channels: block_out_channels[-1], groups: resnet_groups),
        Attention.new(block_out_channels[-1], norm_groups: resnet_groups),
        ResnetBlock2D.new(in_channels: block_out_channels[-1], out_channels: block_out_channels[-1], groups: resnet_groups)
      ]

      channels = block_out_channels.reverse
      channels = [channels[0]] + channels
      self.up_blocks = channels.each_cons(2).each_with_index.map do |(in_ch, out_ch), index|
        EncoderDecoderBlock2D.new(
          in_channels: in_ch,
          out_channels: out_ch,
          num_layers: layers_per_block,
          resnet_groups: resnet_groups,
          add_downsample: false,
          add_upsample: index < block_out_channels.length - 1
        )
      end

      self.conv_norm_out = MLX::NN::GroupNorm.new(resnet_groups, block_out_channels[0], pytorch_compatible: true)
      self.conv_out = MLX::NN::Conv2d.new(block_out_channels[0], out_channels, 3, padding: 1)
    end

    def call(x)
      x = conv_in.call(x)

      x = mid_blocks[0].call(x)
      x = mid_blocks[1].call(x)
      x = mid_blocks[2].call(x)

      up_blocks.each { |layer| x = layer.call(x) }

      x = conv_norm_out.call(x)
      x = MLX::NN.silu(x)
      conv_out.call(x)
    end
  end

  class Autoencoder < MLX::NN::Module
    attr_reader :config, :latent_channels, :scaling_factor

    def initialize(config)
      super()
      @config = config
      @latent_channels = config.latent_channels_in
      @scaling_factor = config.scaling_factor

      self.encoder = Encoder.new(
        in_channels: config.in_channels,
        out_channels: config.latent_channels_out,
        block_out_channels: config.block_out_channels,
        layers_per_block: config.layers_per_block,
        resnet_groups: config.norm_num_groups
      )
      self.decoder = Decoder.new(
        in_channels: config.latent_channels_in,
        out_channels: config.out_channels,
        block_out_channels: config.block_out_channels,
        layers_per_block: config.layers_per_block + 1,
        resnet_groups: config.norm_num_groups
      )

      self.quant_proj = MLX::NN::Linear.new(config.latent_channels_out, config.latent_channels_out)
      self.post_quant_proj = MLX::NN::Linear.new(config.latent_channels_in, config.latent_channels_in)
    end

    def decode(z)
      z = MLX::Core.divide(z, scaling_factor)
      decoder.call(post_quant_proj.call(z))
    end

    def encode(x)
      x = encoder.call(x)
      x = quant_proj.call(x)
      mean, logvar = MLX::Core.split(x, 2, -1)
      mean = MLX::Core.multiply(mean, scaling_factor)
      logvar = MLX::Core.add(logvar, 2 * Math.log(scaling_factor))
      [mean, logvar]
    end

    def call(x)
      mean, logvar = encode(x)
      z = MLX::Core.add(
        MLX::Core.multiply(MLX::Core.random_normal(mean.shape), MLX::Core.exp(MLX::Core.multiply(0.5, logvar))),
        mean
      )
      {
        "x_hat" => decode(z),
        "z" => z,
        "mean" => mean,
        "logvar" => logvar
      }
    end
  end

  module_function

  def download_hf_weights!(destination_dir, repo_id: HF_REPO_ID)
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

  def load_unet_from_hf_directory(model_dir, strict: true, dtype: nil)
    dir = File.expand_path(model_dir.to_s)
    config_path = File.join(dir, "unet", "config.json")
    weights_path = File.join(dir, "unet", "diffusion_pytorch_model.safetensors")

    raise "Missing HF UNet config at #{config_path}" unless File.exist?(config_path)
    raise "Missing HF UNet weights at #{weights_path}" unless File.exist?(weights_path)

    config_payload = JSON.parse(File.binread(config_path))

    block_out_channels = Array(config_payload.fetch("block_out_channels")).map { |v| Integer(v) }
    n_blocks = block_out_channels.length

    layers_per_block = if config_payload["layers_per_block"].is_a?(Array)
      config_payload.fetch("layers_per_block").map { |v| Integer(v) }
    else
      Array.new(n_blocks, Integer(config_payload.fetch("layers_per_block")))
    end

    transformer_layers_per_block = config_payload.fetch("transformer_layers_per_block", Array.new(n_blocks, 1))
    transformer_layers_per_block = Array.new(n_blocks, Integer(transformer_layers_per_block)) unless transformer_layers_per_block.is_a?(Array)
    transformer_layers_per_block = transformer_layers_per_block.map { |v| Integer(v) }

    attention_head_dim = config_payload.fetch("attention_head_dim")
    num_attention_heads = if attention_head_dim.is_a?(Array)
      attention_head_dim.map { |v| Integer(v) }
    else
      Array.new(n_blocks, Integer(attention_head_dim))
    end

    cross_attention_dim = config_payload.fetch("cross_attention_dim")
    cross_attention_dim = if cross_attention_dim.is_a?(Array)
      cross_attention_dim.map { |v| Integer(v) }
    else
      Array.new(n_blocks, Integer(cross_attention_dim))
    end

    config = UNetConfig.new(
      in_channels: Integer(config_payload.fetch("in_channels")),
      out_channels: Integer(config_payload.fetch("out_channels")),
      block_out_channels: block_out_channels,
      layers_per_block: layers_per_block,
      transformer_layers_per_block: transformer_layers_per_block,
      num_attention_heads: num_attention_heads,
      cross_attention_dim: cross_attention_dim,
      norm_num_groups: Integer(config_payload.fetch("norm_num_groups")),
      down_block_types: Array(config_payload.fetch("down_block_types")).map(&:to_s),
      up_block_types: Array(config_payload.fetch("up_block_types")).map(&:to_s).reverse,
      addition_embed_type: config_payload["addition_embed_type"],
      addition_time_embed_dim: config_payload["addition_time_embed_dim"],
      projection_class_embeddings_input_dim: config_payload["projection_class_embeddings_input_dim"]
    )

    model = UNetModel.new(config)

    hf_state, load_note = load_hf_state_from_weights(weights_path, dtype: dtype)
    mapped_state = map_unet_weights(hf_state)
    model.load_weights(mapped_state, strict: strict)
    model.eval
    MLX::Core.eval(model.parameters)

    [model, config_payload, load_note]
  end

  def load_text_encoder_from_hf_directory(model_dir, strict: true, dtype: nil)
    dir = File.expand_path(model_dir.to_s)
    config_path = File.join(dir, "text_encoder", "config.json")
    weights_path = File.join(dir, "text_encoder", "model.safetensors")

    raise "Missing HF text encoder config at #{config_path}" unless File.exist?(config_path)
    raise "Missing HF text encoder weights at #{weights_path}" unless File.exist?(weights_path)

    config_payload = JSON.parse(File.binread(config_path))
    architectures = Array(config_payload["architectures"]).map(&:to_s)
    with_projection = architectures.any? { |name| name.include?("WithProjection") }

    config = CLIPTextModelConfig.new(
      num_layers: Integer(config_payload.fetch("num_hidden_layers")),
      model_dims: Integer(config_payload.fetch("hidden_size")),
      num_heads: Integer(config_payload.fetch("num_attention_heads")),
      max_length: Integer(config_payload.fetch("max_position_embeddings")),
      vocab_size: Integer(config_payload.fetch("vocab_size")),
      projection_dim: with_projection ? Integer(config_payload.fetch("projection_dim")) : nil,
      hidden_act: config_payload.fetch("hidden_act", "quick_gelu").to_s
    )

    model = CLIPTextModel.new(config)

    hf_state, load_note = load_hf_state_from_weights(weights_path, dtype: dtype)
    mapped_state = map_clip_text_encoder_weights(hf_state)
    model.load_weights(mapped_state, strict: strict)
    model.eval
    MLX::Core.eval(model.parameters)

    [model, config_payload, load_note]
  end

  def load_autoencoder_from_hf_directory(model_dir, strict: true, dtype: nil)
    dir = File.expand_path(model_dir.to_s)
    config_path = File.join(dir, "vae", "config.json")
    weights_path = File.join(dir, "vae", "diffusion_pytorch_model.safetensors")

    raise "Missing HF VAE config at #{config_path}" unless File.exist?(config_path)
    raise "Missing HF VAE weights at #{weights_path}" unless File.exist?(weights_path)

    config_payload = JSON.parse(File.binread(config_path))

    config = AutoencoderConfig.new(
      in_channels: Integer(config_payload.fetch("in_channels")),
      out_channels: Integer(config_payload.fetch("out_channels")),
      latent_channels_out: Integer(config_payload.fetch("latent_channels")) * 2,
      latent_channels_in: Integer(config_payload.fetch("latent_channels")),
      block_out_channels: Array(config_payload.fetch("block_out_channels")).map { |v| Integer(v) },
      layers_per_block: Integer(config_payload.fetch("layers_per_block")),
      norm_num_groups: Integer(config_payload.fetch("norm_num_groups")),
      scaling_factor: Float(config_payload.fetch("scaling_factor", 0.18215))
    )

    model = Autoencoder.new(config)

    hf_state, load_note = load_hf_state_from_weights(weights_path, dtype: dtype)
    mapped_state = map_vae_weights(hf_state)
    model.load_weights(mapped_state, strict: strict)
    model.eval
    MLX::Core.eval(model.parameters)

    [model, config_payload, load_note]
  end

  def map_unet_weights(state)
    state.each_with_object({}) do |(key, value), out|
      map_unet_weight_pair(key.to_s, value).each { |mapped_key, mapped_value| out[mapped_key] = mapped_value }
    end
  end

  def map_unet_weight_pair(key, value)
    key = key.dup

    key = key.gsub("downsamplers.0.conv", "downsample") if key.include?("downsamplers")
    key = key.gsub("upsamplers.0.conv", "upsample") if key.include?("upsamplers")

    key = key.gsub("mid_block.resnets.0", "mid_blocks.0") if key.include?("mid_block.resnets.0")
    key = key.gsub("mid_block.attentions.0", "mid_blocks.1") if key.include?("mid_block.attentions.0")
    key = key.gsub("mid_block.resnets.1", "mid_blocks.2") if key.include?("mid_block.resnets.1")

    key = key.gsub("to_k", "key_proj") if key.include?("to_k")
    key = key.gsub("to_out.0", "out_proj") if key.include?("to_out.0")
    key = key.gsub("to_q", "query_proj") if key.include?("to_q")
    key = key.gsub("to_v", "value_proj") if key.include?("to_v")

    if key.include?("ff.net.2")
      key = key.gsub("ff.net.2", "linear3")
    elsif key.include?("ff.net.0")
      key1 = key.gsub("ff.net.0.proj", "linear1")
      key2 = key.gsub("ff.net.0.proj", "linear2")
      value1, value2 = MLX::Core.split(value, 2, 0)
      return [[key1, value1], [key2, value2]]
    end

    value = MLX::Core.squeeze(value) if key.include?("conv_shortcut.weight")
    if value.shape.length == 4 && (key.include?("proj_in") || key.include?("proj_out"))
      value = MLX::Core.squeeze(value)
    end

    if value.shape.length == 4
      value = MLX::Core.transpose(value, [0, 2, 3, 1])
      value = MLX::Core.reshape(value, value.shape)
    end

    [[key, value]]
  end

  def map_clip_text_encoder_weights(state)
    state.each_with_object({}) do |(key, value), out|
      map_clip_text_encoder_weight_pair(key.to_s, value).each { |mapped_key, mapped_value| out[mapped_key] = mapped_value }
    end
  end

  def map_clip_text_encoder_weight_pair(key, value)
    key = key.dup

    key = key[11..] if key.start_with?("text_model.")
    key = key[11..] if key.start_with?("embeddings.")
    key = key[8..] if key.start_with?("encoder.")

    key = key.gsub("self_attn.", "attention.") if key.include?("self_attn.")
    key = key.gsub("q_proj.", "query_proj.") if key.include?("q_proj.")
    key = key.gsub("k_proj.", "key_proj.") if key.include?("k_proj.")
    key = key.gsub("v_proj.", "value_proj.") if key.include?("v_proj.")

    key = key.gsub("mlp.fc1", "linear1") if key.include?("mlp.fc1")
    key = key.gsub("mlp.fc2", "linear2") if key.include?("mlp.fc2")

    [[key, value]]
  end

  def map_vae_weights(state)
    state.each_with_object({}) do |(key, value), out|
      map_vae_weight_pair(key.to_s, value).each { |mapped_key, mapped_value| out[mapped_key] = mapped_value }
    end
  end

  def map_vae_weight_pair(key, value)
    key = key.dup

    key = key.gsub("downsamplers.0.conv", "downsample") if key.include?("downsamplers")
    key = key.gsub("upsamplers.0.conv", "upsample") if key.include?("upsamplers")

    key = key.gsub("to_k", "key_proj") if key.include?("to_k")
    key = key.gsub("to_out.0", "out_proj") if key.include?("to_out.0")
    key = key.gsub("to_q", "query_proj") if key.include?("to_q")
    key = key.gsub("to_v", "value_proj") if key.include?("to_v")

    key = key.gsub("mid_block.resnets.0", "mid_blocks.0") if key.include?("mid_block.resnets.0")
    key = key.gsub("mid_block.attentions.0", "mid_blocks.1") if key.include?("mid_block.attentions.0")
    key = key.gsub("mid_block.resnets.1", "mid_blocks.2") if key.include?("mid_block.resnets.1")

    if key.include?("quant_conv")
      key = key.gsub("quant_conv", "quant_proj")
      value = MLX::Core.squeeze(value)
    end

    value = MLX::Core.squeeze(value) if key.include?("conv_shortcut.weight")

    if value.shape.length == 4
      value = MLX::Core.transpose(value, [0, 2, 3, 1])
      value = MLX::Core.reshape(value, value.shape)
    end

    [[key, value]]
  end

  def normalize_hf_state(raw_state, dtype: nil)
    raw_state.to_a.each_with_object({}) do |(key, value), out|
      tensor = value
      tensor = tensor.astype(dtype) unless dtype.nil?
      out[key.to_s] = tensor
    end
  end

  def load_hf_state_from_weights(weights_path, dtype: nil)
    raw_state = MLX::Core.load(weights_path)
    [normalize_hf_state(raw_state, dtype: dtype), nil]
  rescue StandardError => e
    raise unless safetensors_native_unavailable?(e) && weights_path.end_with?(".safetensors")

    npz_path = ensure_npz_from_safetensors(weights_path)
    raw_state = MLX::Core.load(npz_path)
    [
      normalize_hf_state(raw_state, dtype: dtype),
      "safetensors unavailable; loaded pretrained weights from #{npz_path}"
    ]
  end

  def safetensors_native_unavailable?(error)
    error.message.include?("MLX_BUILD_SAFETENSORS=ON")
  end

  def ensure_npz_from_safetensors(weights_path)
    npz_path = weights_path.sub(/\.safetensors\z/, ".npz")
    if File.exist?(npz_path) && File.size(npz_path).positive? && File.mtime(npz_path) >= File.mtime(weights_path)
      return npz_path
    end

    convert_safetensors_to_npz!(weights_path, npz_path)
    npz_path
  end

  def convert_safetensors_to_npz!(weights_path, npz_path)
    script = <<~PY
      import numpy as np
      from safetensors import safe_open
      import sys

      source = sys.argv[1]
      destination = sys.argv[2]
      tensors = {}

      with safe_open(source, framework="np", device="cpu") as reader:
        keys = list(reader.keys())
        for key in keys:
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
end
