# frozen_string_literal: true

module MLX
  module GraphIR
    module_function

    def load_payload(payload_or_source)
      payload = case payload_or_source
      when Hash
        deep_copy_hash(payload_or_source)
      when String
        if File.file?(payload_or_source)
          parse_json_payload(File.binread(payload_or_source), "graph ir file")
        else
          parse_json_payload(payload_or_source, "graph ir string")
        end
      else
        if payload_or_source.respond_to?(:read)
          parse_json_payload(payload_or_source.read.to_s, "graph ir IO")
        else
          raise TypeError, "graph ir source must be a Hash, JSON String, file path, or IO-like object"
        end
      end
      normalize_hash_keys(payload)
    end

    def dump_json(hash, pretty: true)
      pretty ? JSON.pretty_generate(hash) : JSON.generate(hash)
    end

    def onnx_json_compatible_value(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key] = onnx_json_compatible_value(item)
        end
      when Array
        value.map { |item| onnx_json_compatible_value(item) }
      when ::Complex
        {
          "__mlx_complex__" => [
            value.real.to_f,
            value.imag.to_f
          ]
        }
      when String
        parsed = parse_ruby_complex_literal(value)
        parsed ? { "__mlx_complex__" => parsed } : value
      else
        value
      end
    end

    def normalize_hash_keys(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key.to_s] = normalize_hash_keys(item)
        end
      when Array
        value.map { |item| normalize_hash_keys(item) }
      else
        value
      end
    end
    private_class_method :normalize_hash_keys

    def deep_copy_hash(value)
      case value
      when Hash
        value.each_with_object({}) do |(key, item), out|
          out[key] = deep_copy_hash(item)
        end
      when Array
        value.map { |item| deep_copy_hash(item) }
      else
        value
      end
    end
    private_class_method :deep_copy_hash

    def parse_ruby_complex_literal(value)
      return nil unless value.include?("i")

      complex = begin
        Complex(value)
      rescue ArgumentError, TypeError
        nil
      end
      return nil if complex.nil?

      [complex.real.to_f, complex.imag.to_f]
    end
    private_class_method :parse_ruby_complex_literal

    def parse_json_payload(raw, label)
      JSON.parse(raw)
    rescue JSON::ParserError => e
      raise ArgumentError, "failed to parse #{label}: #{e.message}"
    end
    private_class_method :parse_json_payload
  end
end
