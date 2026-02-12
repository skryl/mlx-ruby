# frozen_string_literal: true

require_relative "test_helper"

class Phase248MemoryControlAccountingParityTest < Minitest::Test
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

  def test_memory_info_controls_and_counters
    old_cache_limit = MLX::Core.set_cache_limit(0)
    a = MLX::Core.zeros([4096], MLX::Core.float32)
    MLX::Core.eval(a)
    a = nil
    GC.start
    assert_equal 0, MLX::Core.get_cache_memory
    assert_equal 0, MLX::Core.set_cache_limit(old_cache_limit)
    assert_equal old_cache_limit, MLX::Core.set_cache_limit(old_cache_limit)

    old_memory_limit = MLX::Core.set_memory_limit(MLX::Core.get_active_memory + 1_000_000)
    previous_limit = MLX::Core.set_memory_limit(old_memory_limit)
    assert_operator previous_limit, :>=, 1_000_000
    assert_equal old_memory_limit, MLX::Core.set_memory_limit(old_memory_limit)

    b = MLX::Core.zeros([4096], MLX::Core.float32)
    MLX::Core.eval(b)
    MLX::Core.synchronize
    active = MLX::Core.get_active_memory
    assert_operator active, :>=, 4096 * 4

    c = MLX::Core.zeros([4096], MLX::Core.float32)
    MLX::Core.eval(c)
    c = nil
    GC.start
    MLX::Core.synchronize
    assert_equal active, MLX::Core.get_active_memory
    assert_operator MLX::Core.get_peak_memory, :>=, 4096 * 8

    MLX::Core.clear_cache
    assert_equal 0, MLX::Core.get_cache_memory
    MLX::Core.reset_peak_memory
    assert_equal 0, MLX::Core.get_peak_memory
  end

  def test_wired_memory_limit_and_active_memory_count_stability
    if MLX::Core.metal_is_available
      max_size = MLX::Core.device_info(MLX::Core.gpu)["max_recommended_working_set_size"]
      assert_raises(RuntimeError) { MLX::Core.set_wired_limit(max_size + 10) }
    end

    MLX::Core.synchronize
    MLX::Core.clear_cache
    init_mem = MLX::Core.get_active_memory

    x = MLX::Core.zeros([128, 128], MLX::Core.float32)
    MLX::Core.eval(x)
    MLX::Core.synchronize
    x = nil
    GC.start

    y = MLX::Core.zeros([90, 128], MLX::Core.float32)
    MLX::Core.eval(y)
    MLX::Core.synchronize
    y = nil
    GC.start
    MLX::Core.synchronize

    final_mem = MLX::Core.get_active_memory
    assert_operator final_mem, :>=, 0
    assert_operator final_mem, :<=, init_mem
  end
end
