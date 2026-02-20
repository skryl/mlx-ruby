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
          native_graph_ir_to_onnx_json(graph_ir_json, opset: opset, model_name: model_name)
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

          native = graph_ir_native_module
          unless native.respond_to?(:export_onnx_json)
            raise RuntimeError,
                  "MLX::GraphIR::Native.export_onnx_json is unavailable; rebuild ext/mlx to a compatible native ABI"
          end

          onnx_json = native.export_onnx_json(fun, extras, trace_kwargs, shapeless, opset, model_name)
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
          external_options = normalize_external_data_options(
            target: target,
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

            if target.respond_to?(:write)
              content = File.binread(onnx_path)
              target.write(content)
              target.rewind if target.respond_to?(:rewind)
              content
            else
              target_path = file_path(target)
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
              nil
            end
          end
        end

        def graph_ir_json_from_payload(payload_or_source)
          payload = MLX::GraphIR.validate!(payload_or_source)
          content = MLX::GraphIR.onnx_json_compatible_value(payload)
          JSON.generate(content)
        end
        private_class_method :graph_ir_json_from_payload

        def native_graph_ir_to_onnx_json(graph_ir_json, opset:, model_name:)
          MLX::Core.ensure_native!
          native = graph_ir_native_module
          unless native.respond_to?(:graph_ir_to_onnx_json)
            raise RuntimeError,
                  "MLX::GraphIR::Native.graph_ir_to_onnx_json is unavailable; rebuild ext/mlx to a compatible native ABI"
          end

          onnx_json = native.graph_ir_to_onnx_json(graph_ir_json, opset, model_name)
          unless onnx_json.is_a?(String)
            raise TypeError, "MLX::GraphIR::Native.graph_ir_to_onnx_json must return String JSON"
          end
          onnx_json
        end
        private_class_method :native_graph_ir_to_onnx_json

        def graph_ir_native_module
          unless defined?(MLX::GraphIR::Native)
            raise RuntimeError,
                  "MLX::GraphIR::Native is unavailable; rebuild ext/mlx to a compatible native ABI"
          end

          MLX::GraphIR::Native
        end
        private_class_method :graph_ir_native_module

        def normalize_external_data_options(
          target:,
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

          if target.respond_to?(:write)
            raise ArgumentError, "external_data export requires a path-like target, not an IO-like target"
          end

          threshold = begin
            Integer(external_data_size_threshold)
          rescue ArgumentError, TypeError
            raise ArgumentError, "external_data_size_threshold must be a non-negative Integer"
          end
          if threshold.negative?
            raise ArgumentError, "external_data_size_threshold must be a non-negative Integer"
          end

          target_path = file_path(target)
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
