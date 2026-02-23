# frozen_string_literal: true

require "json"
require "open3"
require "fileutils"

module MLX
  module ONNX
    module WebGPUHarness
      module_function

      def export_onnx_webgpu_harness(
        target_dir,
        payload_or_source,
        opset: 18,
        model_name: "mlx_graph",
        execution_providers: %w[webgpu wasm],
        benchmark_warmup_runs: 2,
        benchmark_measure_runs: 10,
        external_data: false,
        external_data_size_threshold: 1024,
        external_data_file: nil
      )
        output_dir = file_path(target_dir)
        raise ArgumentError, "target_dir must not be empty" if output_dir.empty?

        providers = normalize_web_execution_providers(execution_providers)
        warmup_runs = normalize_non_negative_integer(
          benchmark_warmup_runs,
          "benchmark_warmup_runs"
        )
        measure_runs = normalize_positive_integer(
          benchmark_measure_runs,
          "benchmark_measure_runs"
        )

        FileUtils.mkdir_p(output_dir)
        model_filename = "model.onnx"
        model_path = File.join(output_dir, model_filename)
        onnx_json = MLX::ONNX.graph_ir_to_onnx_json(
          payload_or_source,
          opset: opset,
          model_name: model_name
        )
        MLX::ONNX.graph_ir_to_onnx(
          model_path,
          payload_or_source,
          opset: opset,
          model_name: model_name,
          external_data: external_data,
          external_data_size_threshold: external_data_size_threshold,
          external_data_file: external_data_file
        )

        stub = JSON.parse(onnx_json)
        input_specs = stub.fetch("graph").fetch("inputs")
        input_examples = build_input_examples(input_specs)

        manifest = {
          "format" => "onnx_webgpu_harness_v1",
          "model" => model_filename,
          "execution_providers" => providers,
          "benchmark" => {
            "warmup_runs" => warmup_runs,
            "measure_runs" => measure_runs
          },
          "inputs" => input_specs.map do |spec|
            {
              "name" => spec.fetch("name"),
              "shape" => spec.fetch("shape"),
              "dtype" => spec.fetch("dtype")
            }
          end
        }
        if external_data
          manifest["external_data"] = [
            external_data_file.nil? ? "model.data" : external_data_file.to_s
          ]
        end

        File.binwrite(
          File.join(output_dir, "harness.manifest.json"),
          JSON.pretty_generate(manifest)
        )
        File.binwrite(
          File.join(output_dir, "inputs.example.json"),
          JSON.pretty_generate(input_examples)
        )
        copy_assets!(output_dir)

        manifest
      end

      def smoke_test_onnx_webgpu_harness(
        harness_dir,
        timeout_seconds: 30,
        mock_ort: false,
        local_ort: true,
        node_bin: ENV.fetch("NODE", "node")
      )
        directory = file_path(harness_dir)
        raise ArgumentError, "harness_dir must not be empty" if directory.empty?

        directory = File.expand_path(directory)
        unless Dir.exist?(directory)
          raise ArgumentError, "harness_dir does not exist: #{directory}"
        end

        timeout = normalize_positive_integer(timeout_seconds, "timeout_seconds")
        mock = normalize_boolean(mock_ort, "mock_ort")
        local = normalize_boolean(local_ort, "local_ort")
        node = node_bin.to_s
        raise ArgumentError, "node_bin must not be empty" if node.empty?

        smoke_script = web_harness_smoke_script_path
        unless File.file?(smoke_script)
          raise RuntimeError, "missing web harness smoke script: #{smoke_script}"
        end

        argv = [
          node,
          smoke_script,
          "--harness-dir",
          directory,
          "--timeout-seconds",
          timeout.to_s
        ]
        argv << "--mock-ort" if mock
        argv << (local ? "--local-ort" : "--no-local-ort")

        stdout, stderr, status = Open3.capture3(*argv, chdir: web_root_dir)
        unless status.success?
          raise RuntimeError, <<~MSG
            web harness smoke test failed: #{argv.join(" ")}
            stdout:
            #{stdout}
            stderr:
            #{stderr}
          MSG
        end

        telemetry = begin
          JSON.parse(stdout)
        rescue JSON::ParserError => e
          raise RuntimeError, <<~MSG
            web harness smoke test produced invalid JSON: #{e.message}
            stdout:
            #{stdout}
            stderr:
            #{stderr}
          MSG
        end

        unless telemetry.is_a?(Hash)
          raise RuntimeError, "web harness smoke test produced non-object telemetry"
        end
        unless telemetry.fetch("format", nil) == "onnx_webgpu_telemetry_v1"
          raise RuntimeError, "unexpected web harness telemetry format: #{telemetry.fetch('format', nil).inspect}"
        end

        telemetry
      end

      def file_path(file)
        if file.respond_to?(:to_path)
          file.to_path.to_s
        else
          file.to_s
        end
      end
      private_class_method :file_path

      def normalize_web_execution_providers(value)
        providers = if value.is_a?(::Array)
          value
        else
          [value]
        end
        providers = providers.map(&:to_s)
        raise ArgumentError, "execution_providers must contain at least one provider" if providers.empty?

        allowed = %w[webgpu wasm]
        providers.each do |provider|
          unless allowed.include?(provider)
            raise ArgumentError, "execution_providers contains unsupported provider #{provider.inspect}"
          end
        end
        providers.uniq
      end
      private_class_method :normalize_web_execution_providers

      def normalize_non_negative_integer(value, label)
        integer = begin
          Integer(value)
        rescue ArgumentError, TypeError
          raise ArgumentError, "#{label} must be a non-negative Integer"
        end
        raise ArgumentError, "#{label} must be a non-negative Integer" if integer.negative?

        integer
      end
      private_class_method :normalize_non_negative_integer

      def normalize_positive_integer(value, label)
        integer = begin
          Integer(value)
        rescue ArgumentError, TypeError
          raise ArgumentError, "#{label} must be a positive Integer"
        end
        raise ArgumentError, "#{label} must be a positive Integer" unless integer.positive?

        integer
      end
      private_class_method :normalize_positive_integer

      def normalize_boolean(value, label)
        unless value == true || value == false
          raise ArgumentError, "#{label} must be true or false"
        end

        value
      end
      private_class_method :normalize_boolean

      def build_input_examples(input_specs)
        input_specs.each_with_object({}) do |spec, out|
          out[spec.fetch("name")] = build_zero_tensor_values(
            spec.fetch("shape"),
            spec.fetch("dtype")
          )
        end
      end
      private_class_method :build_input_examples

      def build_zero_tensor_values(shape, dtype)
        if shape.empty?
          zero_leaf_value_for_dtype(dtype)
        else
          ::Array.new(shape.first) { build_zero_tensor_values(shape[1..], dtype) }
        end
      end
      private_class_method :build_zero_tensor_values

      def zero_leaf_value_for_dtype(dtype)
        if dtype == "bool" || dtype == "bool_"
          false
        elsif dtype == "complex64"
          { "__mlx_complex__" => [0.0, 0.0] }
        elsif dtype.start_with?("float") || dtype == "bfloat16"
          0.0
        else
          0
        end
      end
      private_class_method :zero_leaf_value_for_dtype

      def copy_assets!(output_dir)
        template_dir = web_harness_template_dir
        unless Dir.exist?(template_dir)
          raise RuntimeError, "missing web harness template directory: #{template_dir}"
        end

        %w[index.html harness.js].each do |file_name|
          source = File.join(template_dir, file_name)
          unless File.file?(source)
            raise RuntimeError, "missing web harness template file: #{source}"
          end
          FileUtils.cp(source, File.join(output_dir, file_name))
        end
      end
      private_class_method :copy_assets!

      def web_harness_template_dir
        File.expand_path("../../web/onnx_webgpu_harness", __dir__)
      end
      private_class_method :web_harness_template_dir

      def web_harness_smoke_script_path
        File.join(web_harness_template_dir, "browser_smoke.mjs")
      end
      private_class_method :web_harness_smoke_script_path

      def web_root_dir
        File.expand_path("../../web", __dir__)
      end
      private_class_method :web_root_dir
    end
  end
end
