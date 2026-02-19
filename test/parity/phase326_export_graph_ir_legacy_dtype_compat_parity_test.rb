# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase326ExportGraphIrLegacyDtypeCompatParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_graph_ir_io_target_backfills_legacy_astype_arguments
    payload = legacy_payload_with_missing_astype_arguments
    with_stubbed_native_export_graph_ir(JSON.generate(payload)) do
      io = StringIO.new
      content = MLX::Core.export_graph_ir(io, ->(_x) { nil }, :seed)
      normalized = JSON.parse(content)

      node = normalized.fetch("nodes").find { |entry| entry.fetch("op") == "AsType" }
      assert_equal ["int32"], node.fetch("arguments")
      assert_equal content, io.string
    end
  end

  def test_export_graph_ir_path_target_backfills_legacy_astype_arguments
    payload = legacy_payload_with_missing_astype_arguments
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      with_stubbed_native_export_graph_ir(JSON.generate(payload)) do
        assert_nil MLX::Core.export_graph_ir(path, ->(_x) { nil }, :seed)
      end

      normalized = JSON.parse(File.binread(path))
      node = normalized.fetch("nodes").find { |entry| entry.fetch("op") == "AsType" }
      assert_equal ["int32"], node.fetch("arguments")
    end
  end

  def test_export_graph_ir_keeps_payload_unchanged_when_dtype_argument_is_present
    payload = legacy_payload_with_missing_astype_arguments
    payload.fetch("nodes").first["arguments"] = ["int32"]
    raw_payload = JSON.generate(payload)

    with_stubbed_native_export_graph_ir(raw_payload) do
      content = MLX::Core.export_graph_ir(StringIO.new, ->(_x) { nil }, :seed)
      assert_equal raw_payload, content
    end
  end

  def test_export_graph_ir_path_target_falls_back_when_binread_raises_einval
    payload = legacy_payload_with_missing_astype_arguments
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      with_stubbed_native_export_graph_ir(JSON.generate(payload)) do
        with_stubbed_file_binread_einval(path) do
          assert_nil MLX::Core.export_graph_ir(path, ->(_x) { nil }, :seed)
        end
      end

      normalized = JSON.parse(File.binread(path))
      node = normalized.fetch("nodes").find { |entry| entry.fetch("op") == "AsType" }
      assert_equal ["int32"], node.fetch("arguments")
    end
  end

  private

  def with_stubbed_native_export_graph_ir(raw_payload)
    singleton = MLX::Core.singleton_class
    backup = :__phase326_original_native_export_graph_ir
    singleton.class_eval do
      remove_method(backup) if method_defined?(backup)
      alias_method backup, :native_export_graph_ir
      define_method(:native_export_graph_ir) do |path, *_args|
        File.binwrite(path, raw_payload)
        nil
      end
    end
    yield
  ensure
    singleton.class_eval do
      remove_method :native_export_graph_ir if method_defined?(:native_export_graph_ir)
      if method_defined?(backup)
        alias_method :native_export_graph_ir, backup
        remove_method backup
      end
    end
  end

  def legacy_payload_with_missing_astype_arguments
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

  def with_stubbed_file_binread_einval(target_path)
    singleton = File.singleton_class
    backup = :__phase326_original_file_binread
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
