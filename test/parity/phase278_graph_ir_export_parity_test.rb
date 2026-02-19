# frozen_string_literal: true

require "json"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase278GraphIrExportParityTest < Minitest::Test
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

  def test_export_graph_ir_accepts_file_like_targets
    fun = lambda do |x, y:|
      MLX::Core.add(x, y)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([3.0, 4.0], MLX::Core.float32)

    io = StringIO.new
    written = MLX::Core.export_graph_ir(io, fun, x, y: y)
    payload = JSON.parse(written)

    assert_equal 1, payload.fetch("ir_version")
    assert_equal false, payload.fetch("shapeless")
    assert_equal ["Add"], payload.fetch("nodes").map { |n| n.fetch("op") }
    assert_operator io.string.bytesize, :>, 0
  end

  def test_export_graph_ir_supports_path_targets
    fun = lambda do |x|
      MLX::Core.exp(x)
    end
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      ir_path = File.join(dir, "graph_ir.json")
      assert_nil MLX::Core.export_graph_ir(ir_path, fun, x)

      assert File.exist?(ir_path)
      payload = JSON.parse(File.read(ir_path))
      assert_equal 1, payload.fetch("ir_version")
      assert_equal ["Exp"], payload.fetch("nodes").map { |n| n.fetch("op") }
    end
  end
end
