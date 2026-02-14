# frozen_string_literal: true

require "tmpdir"
require_relative "test_helper"

class Phase251ExportImportAdvancedParityTest < Minitest::Test
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

  def test_export_import_with_variable_inputs_and_control_flow_mask
    fun = lambda do |x, y:, mask:|
      sum = MLX::Core.add(x, y)
      diff = MLX::Core.subtract(x, y)
      [sum, MLX::Core.where(mask, diff, sum)]
    end

    x = MLX::Core.array([3.0, 5.0], MLX::Core.float32)
    y = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    mask = MLX::Core.array([true, false], MLX::Core.bool_)

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "advanced.mlxfn")
      MLX::Core.export_function(path, fun, x, y: y, mask: mask)
      imported = MLX::Core.import_function(path)

      out = imported.call(x, y: y, mask: mask)
      assert_equal 2, out.length
      assert_equal [4.0, 7.0], out[0].to_a
      assert_equal [2.0, 7.0], out[1].to_a
    end
  end

  def test_exporter_multi_signature_and_duplicate_signature_rejection
    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "multi_sig.mlxfn")
      exporter = MLX::Core.exporter(path, ->(z) { [MLX::Core.add(z, 1.0)] })

      x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
      y = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
      exporter.call(x)
      exporter.call(y)
      exporter.close

      imported = MLX::Core.import_function(path)
      assert_equal [2.0, 3.0], imported.call(x).to_a
      assert_equal [2.0, 3.0, 4.0], imported.call(y).to_a
    end

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "dup_sig.mlxfn")
      exporter = MLX::Core.exporter(path, ->(z) { [MLX::Core.add(z, 1.0)] })
      x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
      exporter.call(x)
      err = assert_raises(RuntimeError) { exporter.call(x) }
      assert_match(/same signature/i, err.message)
    end
  end
end
