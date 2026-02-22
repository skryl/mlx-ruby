# frozen_string_literal: true

require "json"
require "tmpdir"
require "fileutils"

require_relative "python_builder"

module MLX
  module GraphIR
    module ONNX
      module Exporter
        module_function

        def graph_ir_to_onnx_json(payload_or_source, opset: 18, model_name: "mlx_graph")
          graph_ir_json = graph_ir_json_from_payload(payload_or_source)
          translate_native_unsupported do
            native_graph_ir_to_onnx_json(graph_ir_json, opset: opset, model_name: model_name)
          end
        end

        def compatibility_report(payload_or_source)
          report_json = compatibility_report_json(payload_or_source)
          report = JSON.parse(report_json)
          unless report.is_a?(Hash)
            raise TypeError, "MLX::GraphIR::Native.graph_ir_compatibility_report_json must return JSON object"
          end
          report
        end

        def compatibility_report_json(payload_or_source)
          graph_ir_json = graph_ir_json_from_payload(payload_or_source)
          native_graph_ir_compatibility_report_json(graph_ir_json)
        end

        def export_onnx_json(
          fun,
          *extras,
          trace_kwargs: {},
          shapeless: false,
          opset: 18,
          model_name: "mlx_graph"
        )
          MLX::Core.ensure_native!
          unless trace_kwargs.is_a?(Hash)
            raise TypeError, "trace_kwargs must be a Hash"
          end
          unless shapeless == true || shapeless == false
            raise TypeError, "shapeless must be true or false"
          end

          onnx_json = translate_native_unsupported do
            MLX::GraphIR::Native.export_onnx_json(fun, extras, trace_kwargs, shapeless, opset, model_name)
          end
          unless onnx_json.is_a?(String)
            raise TypeError, "MLX::GraphIR::Native.export_onnx_json must return String JSON"
          end
          onnx_json
        end

        def onnx_json_to_onnx(
          target,
          onnx_json,
          external_data: false,
          external_data_size_threshold: 1024,
          external_data_file: nil,
          python_bin: ENV.fetch("PYTHON", "python3")
        )
          if target.respond_to?(:write)
            raise ArgumentError, "onnx_json_to_onnx requires a path-like target, not an IO-like target"
          end
          target_path = file_path(target)
          if target_path.empty?
            raise ArgumentError, "onnx_json_to_onnx target must be a non-empty path-like value"
          end

          external_options = normalize_external_data_options(
            target_path: target_path,
            external_data: external_data,
            external_data_size_threshold: external_data_size_threshold,
            external_data_file: external_data_file
          )

          Dir.mktmpdir("mlx-ruby-onnx") do |dir|
            onnx_path = File.join(dir, "graph.onnx")

            PythonBuilder.build_from_stub_json!(
              stub_json: onnx_json,
              output_path: onnx_path,
              external_data: external_options.fetch("enabled"),
              external_data_file: external_options.fetch("file"),
              external_data_size_threshold: external_options.fetch("size_threshold"),
              python_bin: python_bin
            )

            target_dir = File.dirname(target_path)
            FileUtils.mkdir_p(target_dir) unless Dir.exist?(target_dir)
            FileUtils.cp(onnx_path, target_path)

            if external_options.fetch("enabled")
              source = File.join(dir, external_options.fetch("file"))
              if File.file?(source)
                destination = File.join(target_dir, external_options.fetch("file"))
                FileUtils.cp(source, destination)
              end
            end

            target_path
          end
        end

        def graph_ir_json_from_payload(payload_or_source)
          payload = MLX::GraphIR.validate!(payload_or_source)
          content = onnx_json_compatible_value(payload)
          JSON.generate(content)
        end
        private_class_method :graph_ir_json_from_payload

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
        private_class_method :onnx_json_compatible_value

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

        def native_graph_ir_to_onnx_json(graph_ir_json, opset:, model_name:)
          MLX::Core.ensure_native!
          onnx_json = MLX::GraphIR::Native.graph_ir_to_onnx_json(graph_ir_json, opset, model_name)
          unless onnx_json.is_a?(String)
            raise TypeError, "MLX::GraphIR::Native.graph_ir_to_onnx_json must return String JSON"
          end
          onnx_json
        end
        private_class_method :native_graph_ir_to_onnx_json

        def native_graph_ir_compatibility_report_json(graph_ir_json)
          MLX::Core.ensure_native!
          report_json = MLX::GraphIR::Native.graph_ir_compatibility_report_json(graph_ir_json)
          unless report_json.is_a?(String)
            raise TypeError, "MLX::GraphIR::Native.graph_ir_compatibility_report_json must return String JSON"
          end
          report_json
        end
        private_class_method :native_graph_ir_compatibility_report_json

        def translate_native_unsupported
          yield
        rescue MLX::GraphIR::Native::UnsupportedError => e
          raise NotImplementedError, e.message
        end
        private_class_method :translate_native_unsupported

        def normalize_external_data_options(
          target_path:,
          external_data:,
          external_data_size_threshold:,
          external_data_file:
        )
          unless external_data == true || external_data == false
            raise TypeError, "external_data must be true or false"
          end

          unless external_data
            return {
              "enabled" => false,
              "file" => "weights.bin",
              "size_threshold" => 1024
            }
          end

          threshold = begin
            Integer(external_data_size_threshold)
          rescue ArgumentError, TypeError
            raise ArgumentError, "external_data_size_threshold must be a non-negative Integer"
          end
          if threshold.negative?
            raise ArgumentError, "external_data_size_threshold must be a non-negative Integer"
          end

          default_file = begin
            base = File.basename(target_path, File.extname(target_path))
            base = "weights" if base.empty?
            "#{base}.data"
          end
          location = external_data_file.nil? ? default_file : external_data_file.to_s
          if location.empty?
            raise ArgumentError, "external_data_file must be a non-empty filename"
          end
          unless location == File.basename(location)
            raise ArgumentError, "external_data_file must be a filename without path separators"
          end

          {
            "enabled" => true,
            "file" => location,
            "size_threshold" => threshold
          }
        end
        private_class_method :normalize_external_data_options

        def file_path(file)
          if file.respond_to?(:to_path)
            file.to_path.to_s
          else
            file.to_s
          end
        end
        private_class_method :file_path
      end
    end
  end
end
