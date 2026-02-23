# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase326ExportGraphIrNoLegacyNormalizationParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_ir_io_target_keeps_missing_astype_arguments_unchanged
    payload = payload_with_missing_astype_arguments

    with_stubbed_native_export_ir(JSON.generate(payload)) do
      io = StringIO.new
      content = TestSupport.export_graph_ir_to_target(io, ->(_x) { nil }, :seed)
      exported = JSON.parse(content)

      node = exported.fetch("nodes").find { |entry| entry.fetch("op") == "AsType" }
      assert_equal [], node.fetch("arguments")
      assert_equal content, io.string
    end
  end

  def test_export_ir_path_target_keeps_missing_astype_arguments_unchanged
    payload = payload_with_missing_astype_arguments

    Dir.mktmpdir do |dir|
      path = File.join(dir, "ir.json")
      with_stubbed_native_export_ir(JSON.generate(payload)) do
        assert_nil TestSupport.export_graph_ir_to_target(path, ->(_x) { nil }, :seed)
      end

      exported = JSON.parse(File.binread(path))
      node = exported.fetch("nodes").find { |entry| entry.fetch("op") == "AsType" }
      assert_equal [], node.fetch("arguments")
    end
  end

  def test_export_ir_keeps_payload_unchanged_when_dtype_argument_is_present
    payload = payload_with_missing_astype_arguments
    payload.fetch("nodes").first["arguments"] = ["int32"]
    raw_payload = JSON.generate(payload)

    with_stubbed_native_export_ir(raw_payload) do
      content = MLX::ONNX.export_graph_ir_json(->(_x) { nil }, :seed)
      assert_equal raw_payload, content
    end
  end

  private

  def with_stubbed_native_export_ir(raw_payload)
    singleton = MLX::ONNX::Native.singleton_class
    backup_json = :__phase326_original_export_ir_json
    had_export_ir_json =
      singleton.method_defined?(:export_graph_ir_json) ||
      singleton.private_method_defined?(:export_graph_ir_json)

    singleton.class_eval do
      remove_method(backup_json) if method_defined?(backup_json)
      alias_method backup_json, :export_graph_ir_json if had_export_ir_json
      define_method(:export_graph_ir_json) do |*args|
        fun = args.first
        raise TypeError, "expected callable object" unless fun.respond_to?(:call)

        raw_payload
      end
    end
    yield
  ensure
    singleton.class_eval do
      remove_method :export_graph_ir_json if method_defined?(:export_graph_ir_json)
      if method_defined?(backup_json)
        alias_method :export_graph_ir_json, backup_json
        remove_method backup_json
      end
    end
  end

  def payload_with_missing_astype_arguments
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        { "name" => "A", "shape" => [2], "dtype" => "float32" }
      ],
      "keyword_inputs" => [],
      "outputs" => [
        { "name" => "C", "shape" => [2], "dtype" => "int32" }
      ],
      "constants" => [],
      "nodes" => [
        { "op" => "AsType", "inputs" => ["A"], "outputs" => ["B"], "arguments" => [] },
        { "op" => "Reshape", "inputs" => ["B"], "outputs" => ["C"], "arguments" => [[2]] }
      ]
    }
  end
end
