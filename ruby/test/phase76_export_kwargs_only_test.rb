# frozen_string_literal: true

require "tmpdir"
require_relative "test_helper"

class Phase76ExportKwargsOnlyTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_and_import_with_kwargs_only_signature
    fun = lambda do |y:|
      [MLX::Core.add(y, 1.0)]
    end

    y = MLX::Core.array([2.0, 5.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "kw_only.mlxfn")
      MLX::Core.export_function(path, fun, y: y)

      imported = MLX::Core.import_function(path)
      out = imported.call(y: y)
      if out.is_a?(MLX::Core::Array)
        assert_equal [3.0, 6.0], out.to_a
      else
        assert_equal 1, out.length
        assert_equal [3.0, 6.0], out[0].to_a
      end
    end
  end

  def test_exporter_call_with_kwargs_only
    fun = lambda do |y:|
      [MLX::Core.multiply(y, 2.0)]
    end

    y = MLX::Core.array([1.5, 4.0], MLX::Core.float32)

    Dir.mktmpdir do |dir|
      path = File.join(dir, "kw_only_exporter.mlxfn")
      exporter = MLX::Core.exporter(path, fun)
      exporter.call(y: y)
      exporter.close

      imported = MLX::Core.import_function(path)
      out = imported.call(y: y)
      if out.is_a?(MLX::Core::Array)
        assert_equal [3.0, 8.0], out.to_a
      else
        assert_equal 1, out.length
        assert_equal [3.0, 8.0], out[0].to_a
      end
    end
  end
end
