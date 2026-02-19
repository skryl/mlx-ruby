# frozen_string_literal: true

require "json"
require "tmpdir"
require_relative "test_helper"

class Phase329GraphIrFileReadFallbackParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_validate_graph_ir_path_falls_back_when_binread_raises_einval
    payload = graph_ir_payload
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      File.binwrite(path, JSON.generate(payload))

      with_stubbed_file_binread_einval(path) do
        validated = MLX::Core.validate_graph_ir(path)
        assert_equal payload.fetch("outputs"), validated.fetch("outputs")
      end
    end
  end

  def test_graph_ir_to_onnx_stub_path_falls_back_when_binread_raises_einval
    payload = graph_ir_payload
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      File.binwrite(path, JSON.generate(payload))

      with_stubbed_file_binread_einval(path) do
        stub = MLX::Core.graph_ir_to_onnx_stub(path, model_name: "phase329_graph_ir_fallback")
        assert_equal "onnx_stub_v1", stub.fetch("format")
        assert_equal "phase329_graph_ir_fallback", stub.fetch("graph").fetch("name")
      end
    end
  end

  private

  def graph_ir_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        { "name" => "A", "shape" => [2], "dtype" => "float32" }
      ],
      "keyword_inputs" => [],
      "outputs" => [
        { "name" => "C", "shape" => [2], "dtype" => "float32" }
      ],
      "constants" => [
        { "name" => "B", "shape" => [2], "dtype" => "float32", "values" => [1.0, 2.0] }
      ],
      "nodes" => [
        { "op" => "Add", "inputs" => ["A", "B"], "outputs" => ["C"], "arguments" => [] }
      ]
    }
  end

  def with_stubbed_file_binread_einval(target_path)
    singleton = File.singleton_class
    backup = :__phase329_original_file_binread
    singleton.class_eval do
      remove_method(backup) if method_defined?(backup)
      alias_method backup, :binread
      define_method(:binread) do |file, *args|
        path = file.respond_to?(:to_path) ? file.to_path.to_s : file.to_s
        if path == target_path && args.empty?
          raise Errno::EINVAL, path
        end
        send(backup, file, *args)
      end
    end
    yield
  ensure
    singleton.class_eval do
      remove_method :binread if method_defined?(:binread)
      if method_defined?(backup)
        alias_method :binread, backup
        remove_method backup
      end
    end
  end
end
