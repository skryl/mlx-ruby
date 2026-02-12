# frozen_string_literal: true

require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase249GraphExportFilelikeParityTest < Minitest::Test
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

  def test_export_to_dot_accepts_file_like_targets
    a = MLX::Core.array(1.0, MLX::Core.float32)
    b = MLX::Core.array(2.0, MLX::Core.float32)
    c = MLX::Core.add(a, b)

    io = StringIO.new
    written = MLX::Core.export_to_dot(io, c)
    assert_operator written.bytesize, :>, 0
    assert_operator io.string.bytesize, :>, 0
    assert_includes io.string, "digraph"

    multi = MLX::Core.divmod(a, b)
    io2 = StringIO.new
    MLX::Core.export_to_dot(io2, *multi)
    assert_operator io2.string.bytesize, :>, 0
    assert_includes io2.string, "digraph"
  end

  def test_export_to_dot_still_supports_path_targets
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.exp(x)

    Dir.mktmpdir do |dir|
      dot = File.join(dir, "graph.dot")
      assert_nil MLX::Core.export_to_dot(dot, y)
      assert File.exist?(dot)
      assert_includes File.read(dot), "digraph"
    end
  end
end
