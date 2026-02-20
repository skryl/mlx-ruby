# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase331GraphIrExportContractBoundaryParityTest < Minitest::Test
  REQUIRED_TOP_LEVEL_KEYS = %w[
    ir_version
    shapeless
    inputs
    keyword_inputs
    outputs
    constants
    nodes
  ].freeze

  NATIVE_CAPTURE_KEYS = %w[
    shapeless
    inputs
    keyword_inputs
    outputs
    constants
    nodes
  ].freeze

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_graph_ir_contract_matches_between_io_and_path_targets
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    io_payload = export_payload(fun, x, y: y)
    path_payload = export_payload_via_path(fun, x, y: y)

    assert_equal io_payload, path_payload
    assert_equal REQUIRED_TOP_LEVEL_KEYS, io_payload.keys
    assert_equal 1, io_payload.fetch("ir_version")
    assert_equal false, io_payload.fetch("shapeless")
    assert_equal [], io_payload.fetch("constants")
  end

  def test_export_graph_ir_raw_bytes_match_between_io_and_path_targets
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    io = StringIO.new
    io_content = TestSupport.export_graph_ir_to_target(io, fun, x, y: y)
    path_content = export_raw_payload_via_path(fun, x, y: y)

    assert_equal io_content, io.string
    assert_equal io_content, path_content
  end

  def test_native_export_graph_ir_capture_returns_capture_sections_and_ruby_assembly_matches
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    capture = MLX::GraphIR::Native.export_graph_ir_capture(fun, x, y: y)
    exported = export_payload(fun, x, y: y)

    assert_equal NATIVE_CAPTURE_KEYS, capture.keys
    assert_equal false, capture.fetch("shapeless")
    refute capture.key?("ir_version")

    assembled = MLX::GraphIR.assemble_export_payload(capture)
    assert_equal assembled, exported
  end

  def test_export_graph_ir_tensor_info_and_keyword_entry_shapes
    fun = lambda do |x, y:|
      MLX::Core.add(x, y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)
    payload = export_payload(fun, x, y: y)

    (payload.fetch("inputs") + payload.fetch("outputs")).each do |tensor|
      assert_equal %w[name shape dtype], tensor.keys
      assert_kind_of String, tensor.fetch("name")
      assert_operator tensor.fetch("name").length, :>, 0

      shape = tensor.fetch("shape")
      assert_kind_of Array, shape
      assert shape.all? { |dim| dim.is_a?(Integer) && dim >= 0 }

      dtype = tensor.fetch("dtype")
      assert_kind_of String, dtype
      assert_operator dtype.length, :>, 0
    end

    keyword = payload.fetch("keyword_inputs").first
    assert_equal %w[name tensor], keyword.keys
    assert_equal "y", keyword.fetch("name")
    assert_includes payload.fetch("inputs").map { |input| input.fetch("name") }, keyword.fetch("tensor")
  end

  def test_export_graph_ir_node_argument_encoding_shape
    fun = lambda do |x|
      MLX::Core.transpose(MLX::Core.exp(x), [1, 0])
    end
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    payload = export_payload(fun, x)

    payload.fetch("nodes").each do |node|
      assert_equal %w[op inputs outputs arguments], node.keys
      assert_kind_of Array, node.fetch("arguments")
    end

    exp_node = payload.fetch("nodes").find { |node| node.fetch("op") == "Exp" }
    transpose_node = payload.fetch("nodes").find { |node| node.fetch("op") == "Transpose" }

    refute_nil exp_node
    refute_nil transpose_node
    assert_equal [], exp_node.fetch("arguments")
    assert_equal [[1, 0]], transpose_node.fetch("arguments")
  end

  def test_export_graph_ir_constants_include_values_and_node_reference
    fun = lambda do |x|
      bias = MLX::Core.array([[2.0, 3.0], [4.0, 5.0]], MLX::Core.float32)
      MLX::Core.add(x, bias)
    end
    x = MLX::Core.array([[1.0, 1.0], [1.0, 1.0]], MLX::Core.float32)
    payload = export_payload(fun, x)

    constants = payload.fetch("constants")
    assert_equal 1, constants.length

    constant = constants.first
    assert_equal %w[name shape dtype values], constant.keys
    assert_equal [2, 2], constant.fetch("shape")
    assert_equal "float32", constant.fetch("dtype")
    assert_equal [[2.0, 3.0], [4.0, 5.0]], constant.fetch("values")

    constant_name = constant.fetch("name")
    assert payload.fetch("nodes").any? { |node| node.fetch("inputs").include?(constant_name) }
  end

  def test_validate_graph_ir_rejects_unknown_top_level_keys
    fun = lambda do |x|
      MLX::Core.exp(x)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    payload = export_payload(fun, x)
    payload["unknown"] = 1

    error = assert_raises(ArgumentError) do
      MLX::GraphIR.validate!(payload)
    end
    assert_match("unknown keys", error.message)
  end

  private

  def export_payload(fun, *args, **kwargs)
    JSON.parse(MLX::GraphIR.export_graph_ir_json(fun, *args, **kwargs))
  end

  def export_payload_via_path(fun, *args, **kwargs)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      assert_nil TestSupport.export_graph_ir_to_target(path, fun, *args, **kwargs)
      return JSON.parse(File.binread(path))
    end
  end

  def export_raw_payload_via_path(fun, *args, **kwargs)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "graph_ir.json")
      assert_nil TestSupport.export_graph_ir_to_target(path, fun, *args, **kwargs)
      return File.binread(path)
    end
  end
end
