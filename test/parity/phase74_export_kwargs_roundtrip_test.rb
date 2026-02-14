# frozen_string_literal: true

require "tmpdir"
require_relative "test_helper"

class Phase74ExportKwargsRoundtripTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_function_and_import_function_support_kwargs
    fun = lambda do |x, y:|
      [MLX::Core.add(x, y), MLX::Core.subtract(x, y)]
    end

    x = MLX::Core.array([3.0, 5.0], MLX::Core.float32)
    y = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "pair_ops.mlxfn")
      MLX::Core.export_function(path, fun, x, y: y)

      imported = MLX::Core.import_function(path)
      out = imported.call(x, y: y)
      assert_equal 2, out.length
      assert_nested_close [4.0, 7.0], out[0].to_a
      assert_nested_close [2.0, 3.0], out[1].to_a
    end
  end

  def test_exporter_supports_kwargs
    fun = lambda do |x, y:|
      [MLX::Core.multiply(x, y)]
    end

    x = MLX::Core.array([2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.array([4.0, 5.0], MLX::Core.float32)

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "mul_kw.mlxfn")
      exporter = MLX::Core.exporter(path, fun)
      exporter.call(x, y: y)
      exporter.close

      imported = MLX::Core.import_function(path)
      out = imported.call(x, y: y)
      if out.is_a?(MLX::Core::Array)
        assert_nested_close [8.0, 15.0], out.to_a
      else
        assert_equal 1, out.length
        assert_nested_close [8.0, 15.0], out[0].to_a
      end
    end
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
