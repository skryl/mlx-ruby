# frozen_string_literal: true

module MLX
  module NN
    class << self
      def quantize(
        model,
        group_size: nil,
        bits: nil,
        mode: "affine",
        quantize_input: false,
        class_predicate: nil
      )
        class_predicate ||= lambda do |_path, module_obj|
          module_obj.respond_to?(:to_quantized)
        end

        maybe_quantize = lambda do |path, module_obj|
          decision = class_predicate.call(path, module_obj)
          decision = false if decision.respond_to?(:empty?) && decision.empty?

          return module_obj unless decision

          unless module_obj.respond_to?(:to_quantized)
            raise ArgumentError, "Unable to quantize model of type #{module_obj.class}"
          end

          if decision == true
            kwargs = { group_size: group_size, bits: bits, mode: mode }
            kwargs[:quantize_input] = true if quantize_input
            module_obj.to_quantized(**kwargs)
          elsif decision.is_a?(Hash)
            kwargs = symbolize_hash(decision.dup)
            kwargs.delete(:quantize_input) if kwargs[:quantize_input] == false
            module_obj.to_quantized(**kwargs)
          else
            raise ArgumentError, "class_predicate must return a bool or a hash of quantization kwargs"
          end
        end

        leaves = model.leaf_modules
        updated = MLX::Utils.tree_map_with_path(
          maybe_quantize,
          leaves,
          is_leaf: lambda { |v| v.is_a?(MLX::NN::Module) }
        )
        model.update_modules(updated)
        model
      end

      private

      def defaults_for_mode(mode, group_size, bits)
        defaults = {
          "affine" => [64, 4],
          "mxfp4" => [32, 4],
          "nvfp4" => [16, 4],
          "mxfp8" => [32, 8]
        }
        default_group_size, default_bits = defaults.fetch(mode.to_s) do
          raise ArgumentError, "Unsupported quantization mode #{mode}"
        end
        [group_size || default_group_size, bits || default_bits]
      end

      def symbolize_hash(hash)
        hash.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
      end
    end

    class QuantizedEmbedding < Module
      attr_reader :group_size, :bits, :mode, :num_embeddings, :dims

      def initialize(num_embeddings, dims, group_size = nil, bits = nil, mode: "affine")
        super()
        @group_size, @bits = MLX::NN.__send__(:defaults_for_mode, mode, group_size, bits)
        @mode = mode

        scale = Math.sqrt(1.0 / dims)
        weight = MLX::Core.normal([num_embeddings, dims], 0.0, scale)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(weight, group_size, bits, mode)
        self.weight = q_weight
        self.scales = q_scales
        self.biases = q_biases.empty? ? nil : q_biases[0]

        @num_embeddings = num_embeddings
        @dims = dims
        freeze
      end

      def call(x)
        gathered_weight = MLX::Core.take(weight, x, 0)
        gathered_scales = MLX::Core.take(scales, x, 0)
        gathered_biases = biases.nil? ? nil : MLX::Core.take(biases, x, 0)
        MLX::Core.dequantize(gathered_weight, gathered_scales, gathered_biases, group_size, bits, mode)
      end

      def as_linear(x)
        MLX::Core.quantized_matmul(x, weight, scales, biases, true, group_size, bits, mode)
      end

      def self.from_embedding(embedding_layer, group_size = nil, bits = nil, mode: "affine")
        num_embeddings, dims = embedding_layer.weight.shape
        out = new(num_embeddings, dims, group_size, bits, mode: mode)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(embedding_layer.weight, group_size, bits, mode)
        out.weight = q_weight
        out.scales = q_scales
        out.biases = q_biases.empty? ? nil : q_biases[0]
        out
      end
    end

    class QuantizedLinear < Module
      attr_reader :group_size, :bits, :mode

      def initialize(input_dims, output_dims, bias = true, group_size = nil, bits = nil, mode: "affine")
        super()
        @group_size, @bits = MLX::NN.__send__(:defaults_for_mode, mode, group_size, bits)
        @mode = mode

        scale = Math.sqrt(1.0 / input_dims)
        weight = MLX::Core.uniform([output_dims, input_dims], -scale, scale)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(weight, group_size, bits, mode)
        self.weight = q_weight
        self.scales = q_scales
        self.biases = q_biases.empty? ? nil : q_biases[0]
        self.bias = MLX::Core.zeros([output_dims], MLX::Core.float32) if bias

        freeze
      end

      def call(x)
        out = MLX::Core.quantized_matmul(x, weight, scales, biases, true, group_size, bits, mode)
        if state.key?("bias")
          MLX::Core.add(out, bias)
        else
          out
        end
      end

      def self.from_linear(linear_layer, group_size = nil, bits = nil, mode: "affine")
        output_dims, input_dims = linear_layer.weight.shape
        out = new(input_dims, output_dims, false, group_size, bits, mode: mode)
        q_weight, q_scales, *q_biases = MLX::Core.quantize(linear_layer.weight, group_size, bits, mode)
        out.weight = q_weight
        out.scales = q_scales
        out.biases = q_biases.empty? ? nil : q_biases[0]
        out.bias = linear_layer.bias if linear_layer.state.key?("bias")
        out
      end
    end

    class QQLinear < Module
      attr_reader :group_size, :bits, :mode

      def initialize(input_dims, output_dims, group_size = nil, bits = nil, mode: "nvfp4")
        super()
        @group_size, @bits = MLX::NN.__send__(:defaults_for_mode, mode, group_size, bits)
        @mode = mode

        scale = Math.sqrt(1.0 / input_dims)
        self.weight = MLX::Core.uniform([output_dims, input_dims], -scale, scale)
        @_quantized = false
      end

      def quantize
        return self if @_quantized

        q_weight, q_scales, *_ = MLX::Core.quantize(weight, group_size, bits, mode)
        self.weight = q_weight
        self.scales = q_scales
        @_quantized = true
        self
      end

      def dequantize
        return self unless @_quantized

        self.weight = MLX::Core.dequantize(weight, scales, nil, group_size, bits, mode)
        state.delete("scales")
        @_quantized = false
        self
      end

      def train(mode = true)
        super
        if training
          dequantize
        else
          quantize
        end
        self
      end

      def call(x)
        q_scales = state["scales"]
        MLX::Core.qqmm(x, weight, q_scales, group_size, bits, mode)
      end

      def self.from_linear(linear_layer, group_size = nil, bits = nil, mode: "nvfp4")
        if linear_layer.state.key?("bias")
          raise NotImplementedError, "QQLinear does not support bias yet."
        end

        output_dims, input_dims = linear_layer.weight.shape
        out = new(input_dims, output_dims, group_size, bits, mode: mode)
        out.weight = linear_layer.weight
        out.train(linear_layer.training)
        out
      end
    end
  end
end
