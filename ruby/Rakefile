# frozen_string_literal: true

require "rake/testtask"
require_relative "tasks/benchmark_task"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = true
end

namespace :benchmark do
  def self.options
    raw_device = ENV.fetch("DEVICE", "gpu").downcase
    compute_device = raw_device == "metal" ? "gpu" : raw_device
    unless %w[cpu gpu].include?(compute_device)
      raise "Invalid DEVICE='#{raw_device}'. Use cpu or gpu."
    end

    {
      iterations: ENV.fetch("ITERATIONS", BenchmarkTask::DEFAULT_ITERATIONS).to_i,
      warmup: ENV.fetch("WARMUP", BenchmarkTask::DEFAULT_WARMUP).to_i,
      batch_size: ENV.fetch("BATCH", BenchmarkTask::DEFAULT_BATCH_SIZE).to_i,
      sequence_length: ENV.fetch("SEQUENCE_LENGTH", BenchmarkTask::DEFAULT_SEQUENCE_LENGTH).to_i,
      target_sequence_length: ENV.fetch("TARGET_SEQUENCE_LENGTH", BenchmarkTask::DEFAULT_TARGET_SEQUENCE_LENGTH).to_i,
      dims: ENV.fetch("DIMENSIONS", BenchmarkTask::DEFAULT_DIMS).to_i,
      num_heads: ENV.fetch("HEADS", BenchmarkTask::DEFAULT_HEADS).to_i,
      num_layers: ENV.fetch("LAYERS", BenchmarkTask::DEFAULT_LAYERS).to_i,
      compute_device: compute_device,
      python_bin: ENV.fetch("PYTHON", "python3")
    }
  end

  desc "Compare Ruby and Python transformer implementations."
  task :transformer do
    task = BenchmarkTask.new(**options)
    task.run(model: :transformer)
  end

  desc "Compare Ruby and Python CNN implementations."
  task :cnn do
    task = BenchmarkTask.new(**options)
    task.run(model: :cnn)
  end

  desc "Compare Ruby and Python MLP implementations."
  task :mlp do
    task = BenchmarkTask.new(**options)
    task.run(model: :mlp)
  end

  desc "Compare Ruby and Python RNN implementations."
  task :rnn do
    task = BenchmarkTask.new(**options)
    task.run(model: :rnn)
  end

  desc "Compare Ruby and Python GPT-2 implementation (Karpathy tiny-shakespeare full training loop)."
  task :karpathy_gpt2 do
    task = BenchmarkTask.new(**options)
    task.run(model: :karpathy_gpt2)
  end

  desc "Run all configured benchmarks (transformer, cnn, mlp, rnn, karpathy_gpt2)."
  task all: %i[transformer cnn mlp rnn karpathy_gpt2]
end

task default: :test
