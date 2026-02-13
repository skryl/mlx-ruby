# frozen_string_literal: true

require_relative "test_helper"

class Phase149SchedulerFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_join_schedules_uses_step_offset_after_boundary
    warmup = MLX::Optimizers.linear_schedule(0.0, 1.0, 10)
    decay = MLX::Optimizers.cosine_decay(1.0, 10, 0.0)
    schedule = MLX::Optimizers.join_schedules([warmup, decay], [10])

    assert_in_delta 0.0, schedule.call(0), 1e-8
    assert_in_delta 0.9, schedule.call(9), 1e-8

    # Python semantics: second schedule receives (step - boundary).
    assert_in_delta 1.0, schedule.call(10), 1e-8
    assert_in_delta decay.call(2), schedule.call(12), 1e-8
  end

  def test_join_schedules_validates_input
    err = assert_raises(ArgumentError) do
      MLX::Optimizers.join_schedules([], [])
    end
    assert_match(/at least 1 schedule/i, err.message)

    err = assert_raises(ArgumentError) do
      MLX::Optimizers.join_schedules([->(_step) { 0.0 }], [1])
    end
    assert_match(/boundar/i, err.message)
  end

  def test_linear_schedule_validates_steps
    err = assert_raises(ArgumentError) do
      MLX::Optimizers.linear_schedule(0.0, 1.0, 0)
    end
    assert_match(/steps/i, err.message)
  end
end
