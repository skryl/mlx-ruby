# frozen_string_literal: true

require_relative "test_helper"

class Phase50IoFormatsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_save_safetensors_roundtrip_or_feature_error
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "weights.safetensors")

      begin
        MLX::Core.save_safetensors(path, { "x" => x }, { "note" => "ok" })
        arrays, metadata = MLX::Core.load(path, "safetensors", true)
        assert MLX::Core.array_equal(x, arrays["x"])
        assert_equal "ok", metadata["note"] if metadata.key?("note")
      rescue RuntimeError => e
        assert_match(/SAFETENSORS|safetensors/i, e.message)
      end
    end
  end

  def test_save_gguf_roundtrip_or_feature_error
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    TestSupport.mktmpdir do |dir|
      path = File.join(dir, "weights.gguf")

      begin
        MLX::Core.save_gguf(path, { "x" => x }, { "author" => "ruby" })
        arrays, metadata = MLX::Core.load(path, "gguf", true)
        assert arrays.key?("x")
        assert_equal "ruby", metadata["author"] if metadata.key?("author")
      rescue RuntimeError => e
        assert_match(/GGUF|gguf/i, e.message)
      end
    end
  end

  def test_savez_methods_roundtrip
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    TestSupport.mktmpdir do |dir|
      uncompressed = File.join(dir, "weights")
      MLX::Core.savez(uncompressed, x: x)
      loaded = MLX::Core.load(uncompressed + ".npz")
      assert MLX::Core.array_equal(x, loaded["x"])

      compressed = File.join(dir, "weights_compressed.npz")
      MLX::Core.savez_compressed(compressed, x: x)
      loaded2 = MLX::Core.load(compressed)
      assert MLX::Core.array_equal(x, loaded2["x"])
    end
  end
end
