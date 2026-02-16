# frozen_string_literal: true

require "digest"

module BenchmarkExamples
  module BenchmarkDigest
    INPUT_BASE_OFFSET = 1_000_000

    module_function

    def deterministic_tensor(shape, dtype, offset: 0)
      flat_count = shape.inject(1, :*)
      dtype_name = dtype.to_s

      if dtype_name.include?("int")
        values = Array.new(flat_count) { |index| (offset + index) % 97 }
        array = MLX::Core.array(values, MLX::Core.int32)
      else
        values = Array.new(flat_count) do |index|
          (((offset + index) % 1024) - 512) / 257.0
        end
        array = MLX::Core.array(values, MLX::Core.float32)
      end

      array = array.astype(dtype) unless array.dtype == dtype
      MLX::Core.reshape(array, shape)
    end

    def assign_deterministic_parameters!(module_or_modules)
      modules = module_or_modules.is_a?(Array) ? module_or_modules : [module_or_modules]
      offset = 0

      modules.each do |module_obj|
        parameters = module_obj.parameters
        deterministic_parameters, offset = deterministic_tree(parameters, offset)
        module_obj.update(deterministic_parameters)
      end
    end

    def digest_value(value)
      Digest::SHA256.hexdigest(encode_value(value))
    end

    def digest_array(array_like)
      digest_value(array_like.to_a)
    end

    def encode_value(value)
      case value
      when Integer
        +"i#{[value].pack('q<')}"
      when Float
        +"f#{[value].pack('E')}"
      when true
        +"b\x01"
      when false
        +"b\x00"
      when NilClass
        +"n"
      when String
        data = value.b
        +"s#{[data.bytesize].pack('L<')}#{data}"
      when Array
        encoded = +"a#{[value.length].pack('L<')}"
        value.each { |element| encoded << encode_value(element) }
        encoded
      when Hash
        pairs = value.map { |key, child| [key.to_s, child] }.sort_by(&:first)
        encoded = +"h#{[pairs.length].pack('L<')}"
        pairs.each do |key, child|
          encoded << encode_value(key)
          encoded << encode_value(child)
        end
        encoded
      else
        if value.respond_to?(:to_a)
          encode_value(value.to_a)
        else
          raise ArgumentError, "Unsupported value for digest encoding: #{value.class}"
        end
      end
    end

    def deterministic_tree(tree, offset)
      case tree
      when Hash
        out = {}
        tree.keys.sort_by(&:to_s).each do |key|
          child, offset = deterministic_tree(tree.fetch(key), offset)
          out[key] = child
        end
        [out, offset]
      when Array
        out = []
        tree.each do |element|
          child, offset = deterministic_tree(element, offset)
          out << child
        end
        [out, offset]
      when NilClass
        [nil, offset]
      else
        unless tree.respond_to?(:shape) && tree.respond_to?(:dtype)
          raise ArgumentError, "Unsupported parameter leaf for deterministic tree: #{tree.class}"
        end

        shape = tree.shape
        flat_count = shape.inject(1, :*)
        [deterministic_tensor(shape, tree.dtype, offset: offset), offset + flat_count]
      end
    end
  end
end
