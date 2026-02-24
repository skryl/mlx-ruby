# frozen_string_literal: true

require_relative "test_helper"
require_relative "../../../tasks/benchmark_task"
require_relative "../../../tasks/examples_models_benchmark_adapter"

class Phase326WebgpuPythonParityMetricsTest < Minitest::Test
  def setup
    @benchmark_task = BenchmarkTask.new(
      iterations: 1,
      warmup: 0,
      batch_size: 1,
      sequence_length: 2,
      target_sequence_length: 2,
      dims: 4,
      num_heads: 2,
      num_layers: 1,
      compute_device: :cpu
    )

    @adapter = ExamplesModelsBenchmarkAdapter.new(
      repo_root: RUBY_ROOT,
      submodule_root: File.join(RUBY_ROOT, "submodules", "mlx-ruby-examples"),
      device: :cpu,
      runs: 1,
      warmup: 0,
      timeout: 1,
      mode: :dsl
    )
  end

  def test_benchmark_task_webgpu_parity_metrics
    metrics = @benchmark_task.send(
      :webgpu_parity_metrics,
      expected: [0.0, 1.0, -1.0],
      actual: [0.25, 0.5, -1.0]
    )

    assert_in_delta 0.5, metrics.fetch("max_diff"), 1e-9
    assert_equal true, metrics.fetch("ok")
    assert_operator metrics.fetch("tolerance"), :>=, 1.0
  end

  def test_examples_adapter_output_map_parity_metrics
    metrics = @adapter.send(
      :output_map_parity_metrics,
      output_names: ["logits"],
      expected_outputs: { "logits" => [1.0, 2.0, 3.0] },
      sample_outputs: { "output_0" => [1.0, 2.0, 3.6] }
    )

    assert_in_delta 0.6, metrics.fetch("max_diff"), 1e-9
    assert_equal true, metrics.fetch("ok")
    assert_operator metrics.fetch("tolerance"), :>=, 1.0
  end
end
