# frozen_string_literal: true

require "tempfile"
require_relative "../support/test_helper"
require_relative "../scripts/profile_test_timings"

class TestTimingProfilerTest < Minitest::Test
  def test_parse_verbose_log_extracts_test_method_durations
    Tempfile.create(["timings", ".log"]) do |log|
      log.write(<<~LOG)
        Run options: --verbose --seed 1234

        # Running:
        FooTest#test_fast_case = 0.12 s = .
        FooTest#test_skipped_case = 0.00 s = S
        Bar::BazTest#test_slow_case = 31.45 s = .

        Finished in 31.570000s, 0.0950 runs/s, 0.0950 assertions/s.
      LOG
      log.flush

      parsed = TestTimingProfiler.parse_verbose_log(log.path)
      assert_in_delta 0.12, parsed.fetch("FooTest#test_fast_case"), 1e-9
      assert_in_delta 0.0, parsed.fetch("FooTest#test_skipped_case"), 1e-9
      assert_in_delta 31.45, parsed.fetch("Bar::BazTest#test_slow_case"), 1e-9
    end
  end

  def test_merge_device_timings_uses_max_cpu_gpu_duration
    merged = TestTimingProfiler.merge_device_timings(
      {
        "FooTest#test_fast_case" => 0.12,
        "BarTest#test_cpu_only" => 1.0,
        "SharedTest#test_case" => 8.0
      },
      {
        "BazTest#test_gpu_only" => 2.0,
        "SharedTest#test_case" => 42.5
      }
    )

    assert_equal 4, merged.keys.length
    assert_equal 0.12, merged.fetch("FooTest#test_fast_case").fetch("cpu_seconds")
    assert_nil merged.fetch("FooTest#test_fast_case")["gpu_seconds"]
    assert_equal 0.12, merged.fetch("FooTest#test_fast_case").fetch("max_seconds")

    assert_nil merged.fetch("BazTest#test_gpu_only")["cpu_seconds"]
    assert_equal 2.0, merged.fetch("BazTest#test_gpu_only").fetch("gpu_seconds")
    assert_equal 2.0, merged.fetch("BazTest#test_gpu_only").fetch("max_seconds")

    assert_equal 8.0, merged.fetch("SharedTest#test_case").fetch("cpu_seconds")
    assert_equal 42.5, merged.fetch("SharedTest#test_case").fetch("gpu_seconds")
    assert_equal 42.5, merged.fetch("SharedTest#test_case").fetch("max_seconds")
  end

  def test_build_slow_registry_filters_tests_above_threshold
    merged = {
      "FastTest#test_a" => {"cpu_seconds" => 1.2, "gpu_seconds" => 0.7, "max_seconds" => 1.2},
      "BoundaryTest#test_b" => {"cpu_seconds" => 30.0, "gpu_seconds" => 29.5, "max_seconds" => 30.0},
      "SlowTest#test_c" => {"cpu_seconds" => 10.0, "gpu_seconds" => 31.0, "max_seconds" => 31.0}
    }

    registry = TestTimingProfiler.build_slow_registry(merged, threshold_seconds: 30.0)
    assert_equal "mlx_slow_tests_v1", registry.fetch("format")
    assert_equal 30.0, registry.fetch("threshold_seconds")
    assert_equal ["SlowTest#test_c"], registry.fetch("tests").keys
  end
end
