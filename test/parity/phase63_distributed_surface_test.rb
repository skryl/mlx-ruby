# frozen_string_literal: true

require_relative "test_helper"

class Phase63DistributedSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_distributed_group_and_collectives
    assert_respond_to MLX::Core, :distributed_is_available
    assert_respond_to MLX::Core, :init
    assert_respond_to MLX::Core, :all_sum
    assert_respond_to MLX::Core, :all_max
    assert_respond_to MLX::Core, :all_min
    assert_respond_to MLX::Core, :all_gather
    assert_respond_to MLX::Core, :sum_scatter
    assert_respond_to MLX::Core, :send
    assert_respond_to MLX::Core, :recv
    assert_respond_to MLX::Core, :recv_like

    avail = MLX::Core.distributed_is_available
    assert_includes [true, false], avail

    group = MLX::Core.init(false, "any")
    assert_instance_of MLX::Core::Group, group
    assert_operator group.rank, :>=, 0
    assert_operator group.size, :>=, 1

    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)

    all_sum = MLX::Core.all_sum(x, group)
    all_max = MLX::Core.all_max(x, group)
    all_min = MLX::Core.all_min(x, group)
    gathered = MLX::Core.all_gather(x, group)
    scattered = MLX::Core.sum_scatter(x, group)

    if group.size == 1
      assert_equal [1.0, 2.0, 3.0], all_sum.to_a
      assert_equal [1.0, 2.0, 3.0], all_max.to_a
      assert_equal [1.0, 2.0, 3.0], all_min.to_a
      assert_equal [1.0, 2.0, 3.0], gathered.to_a
      assert_equal [1.0, 2.0, 3.0], scattered.to_a
    else
      assert_equal [3], all_sum.shape
      assert_equal [3], all_max.shape
      assert_equal [3], all_min.shape
      assert_operator gathered.shape[0], :>=, 3
      assert_equal [3], scattered.shape
    end
  end
end
