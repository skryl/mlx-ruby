# frozen_string_literal: true

require "open3"
require "tempfile"
require "tmpdir"
require_relative "test_helper"

class Phase250LoadSaveEdgeParityTest < Minitest::Test
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

  def test_load_npy_dtype_and_complex_error
    Dir.mktmpdir do |dir|
      file = File.join(dir, "tensor.npy")
      begin
        run_python!("python3", "-c", "import numpy as np,sys;a=np.random.randn(8).astype(np.float64);np.save(sys.argv[1],a)", file)
      rescue RuntimeError => e
        skip("numpy unavailable: #{e.message}") if e.message =~ /No module named 'numpy'/
        raise
      end

      out = MLX::Core.load(file)
      assert_equal MLX::Core.float64, out.dtype
      assert_equal 8, out.shape[0]

      run_python!("python3", "-c", "import numpy as np,sys;a=np.random.randn(8).astype(np.float64);b=np.random.randn(8).astype(np.float64);np.save(sys.argv[1],a+0j*b)", file)
      err = assert_raises(RuntimeError) { MLX::Core.load(file) }
      assert_match(/Unsupported array protocol type-string/i, err.message)
    end
  end

  def test_non_contiguous_roundtrip_and_optional_container_formats
    a = MLX::Core.broadcast_to(MLX::Core.array([1, 2], MLX::Core.int32), [4, 2])
    Dir.mktmpdir do |dir|
      npy = File.join(dir, "a.npy")
      MLX::Core.save(npy, a)
      assert MLX::Core.array_equal(a, MLX::Core.load(npy))

      transposed = MLX::Core.swapaxes(MLX::Core.reshape(MLX::Core.arange(0, 4, 1, MLX::Core.int32), [2, 2]), 0, 1)

      safetensors = File.join(dir, "a.safetensors")
      begin
        MLX::Core.save_safetensors(safetensors, {"a" => transposed})
        assert MLX::Core.array_equal(transposed, MLX::Core.load(safetensors)["a"])
      rescue RuntimeError => e
        assert_match(/SAFETENSORS|safetensors/i, e.message)
      end

      gguf = File.join(dir, "a.gguf")
      begin
        MLX::Core.save_gguf(gguf, {"a" => transposed})
        assert MLX::Core.array_equal(transposed, MLX::Core.load(gguf)["a"])
      rescue RuntimeError => e
        assert_match(/GGUF|gguf/i, e.message)
      end
    end
  end

  def test_gguf_metadata_and_fp8_load_path_or_feature_error
    Dir.mktmpdir do |dir|
      gguf = File.join(dir, "meta.gguf")
      tensor = MLX::Core.ones([2, 2], MLX::Core.int32)
      begin
        MLX::Core.save_gguf(gguf, {"test" => tensor}, {"meta" => "data"})
        arrays, metadata = MLX::Core.load(gguf, "gguf", true)
        assert MLX::Core.array_equal(tensor, arrays["test"])
        assert_equal "data", metadata["meta"] if metadata.key?("meta")
      rescue RuntimeError => e
        assert_match(/GGUF|gguf/i, e.message)
      end
    end

    contents = "H\x00\x00\x00\x00\x00\x00\x00{\"tensor\":{\"dtype\":\"F8_E4M3\",\"shape\":[10],\"data_offsets\":[0,10]}}       \x00~\xFE\xB6.\x83\xBA\xBA\xBC\x82".b
    Tempfile.create(["f8", ".safetensors"]) do |f|
      f.write(contents)
      f.flush

      begin
        out = MLX::Core.load(f.path)["tensor"]
        fp = MLX::Core.from_fp8(out)
        assert_equal [10], fp.shape
      rescue RuntimeError => e
        assert_match(/SAFETENSORS|safetensors|fp8/i, e.message)
      end
    end
  end

  private

  def run_python!(*cmd)
    stdout, stderr, status = Open3.capture3(*cmd)
    return if status.success?

    raise <<~MSG
      python command failed: #{cmd.join(" ")}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end
end
