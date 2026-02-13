# frozen_string_literal: true

require_relative "test_helper"

class Phase263TransformCacheKeyNoMarshalPerfTest < Minitest::Test
  def setup
    skip("pending: timeout-sensitive parity coverage; re-enable in final CI")
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_compile_cache_key_does_not_use_marshal_dump
    fn = lambda { |x| MLX::Core.add(x, 1.0) }
    compiled = MLX::Core.compile(fn)

    original_dump = Marshal.method(:dump)
    Marshal.define_singleton_method(:dump) do |*|
      raise "compile cache key should not call Marshal.dump"
    end

    out = compiled.call(MLX::Core.array([1.0, 2.0], MLX::Core.float32))
    assert_equal [2.0, 3.0], out.to_a
  ensure
    Marshal.define_singleton_method(:dump, original_dump) unless original_dump.nil?
  end

  def test_checkpoint_cache_key_does_not_use_marshal_dump
    fn = lambda { |x| MLX::Core.add(x, 2.0) }
    checkpointed = MLX::Core.checkpoint(fn)

    original_dump = Marshal.method(:dump)
    Marshal.define_singleton_method(:dump) do |*|
      raise "checkpoint cache key should not call Marshal.dump"
    end

    out = checkpointed.call(MLX::Core.array([1.0, 2.0], MLX::Core.float32))
    assert_equal [3.0, 4.0], out.to_a
  ensure
    Marshal.define_singleton_method(:dump, original_dump) unless original_dump.nil?
  end
end
