# frozen_string_literal: true

require_relative "test_helper"

class Phase2CoreNativeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_native_core_version_and_memory_entrypoints
    assert MLX.native_available?, "native extension should be available"
    assert defined?(MLX::Core)

    assert_equal MLX::VERSION, MLX::Core.version

    assert_respond_to MLX::Core, :get_active_memory
    assert_respond_to MLX::Core, :get_peak_memory
    assert_respond_to MLX::Core, :reset_peak_memory
    assert_respond_to MLX::Core, :get_cache_memory
    assert_respond_to MLX::Core, :set_memory_limit
    assert_respond_to MLX::Core, :set_cache_limit
    assert_respond_to MLX::Core, :set_wired_limit
    assert_respond_to MLX::Core, :clear_cache

    assert_kind_of Integer, MLX::Core.get_active_memory
    assert_kind_of Integer, MLX::Core.get_peak_memory
    assert_kind_of Integer, MLX::Core.get_cache_memory

    prev_memory_limit = MLX::Core.set_memory_limit(MLX::Core.get_active_memory + 1_073_741_824)
    assert_kind_of Integer, prev_memory_limit
    MLX::Core.set_memory_limit(prev_memory_limit)
  end

  def test_native_device_and_stream_entrypoints
    assert defined?(MLX::Core::Device)
    assert defined?(MLX::Core::Stream)

    cpu = MLX::Core::Device.new(:cpu, 0)
    assert_equal :cpu, cpu.type
    assert_equal 0, cpu.index

    default_device = MLX::Core.default_device
    assert_instance_of MLX::Core::Device, default_device

    assert_respond_to MLX::Core, :set_default_device
    MLX::Core.set_default_device(default_device)

    assert_respond_to MLX::Core, :is_available
    assert_includes [true, false], MLX::Core.is_available(default_device)
    assert MLX::Core.is_available(:cpu)

    assert_respond_to MLX::Core, :device_count
    assert_operator MLX::Core.device_count(:cpu), :>=, 1

    assert_respond_to MLX::Core, :device_info
    info = MLX::Core.device_info(default_device)
    assert_kind_of Hash, info
    refute_empty info

    assert_respond_to MLX::Core, :default_stream
    assert_respond_to MLX::Core, :set_default_stream
    assert_respond_to MLX::Core, :new_stream
    assert_respond_to MLX::Core, :synchronize

    default_stream = MLX::Core.default_stream(default_device)
    assert_instance_of MLX::Core::Stream, default_stream
    MLX::Core.set_default_stream(default_stream)

    stream = MLX::Core.new_stream(default_device)
    assert_instance_of MLX::Core::Stream, stream

    MLX::Core.synchronize
    MLX::Core.synchronize(stream)
  end
end
