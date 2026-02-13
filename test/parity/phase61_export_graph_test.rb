# frozen_string_literal: true

require "tmpdir"
require_relative "test_helper"

class Phase61ExportGraphTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_function_and_import_function_roundtrip
    fun = ->(x) { MLX::Core.add(x, 2.0) }
    x = MLX::Core.array([1.0, 3.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "add2.mlxfn")
      MLX::Core.export_function(path, fun, [x])

      imported = MLX::Core.import_function(path)
      out = imported.call(x)
      assert_equal [3.0, 5.0], out.to_a
    end
  end

  def test_exporter_and_export_to_dot
    fun = ->(x) { MLX::Core.multiply(x, 3.0) }
    x = MLX::Core.array([2.0, 4.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      fn_path = File.join(dir, "mul3.mlxfn")
      exporter = MLX::Core.exporter(fn_path, fun)
      exporter.call(x)
      exporter.close

      imported = MLX::Core.import_function(fn_path)
      assert_equal [6.0, 12.0], imported.call(x).to_a

      dot_path = File.join(dir, "graph.dot")
      y = MLX::Core.exp(x)
      MLX::Core.export_to_dot(dot_path, y)
      assert File.exist?(dot_path)
      content = File.read(dot_path)
      assert_includes content, "digraph"
    end
  end
end
