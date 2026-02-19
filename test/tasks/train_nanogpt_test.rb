# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../tasks/training_task/train_nanogpt_shakespeare"

class WebTrainNanogptShakespeareOptionsTest < Minitest::Test
  ENV_KEYS = %w[
    NANOGPT_DEVICE
    NANOGPT_BATCH_SIZE
    NANOGPT_MICRO_BATCH_SIZE
    NANOGPT_GRAD_ACCUM
    NANOGPT_EVAL_BATCH_SIZE
    NANOGPT_UNSAFE_BATCHING
    NANOGPT_GRAD_CHECKPOINT
  ].freeze

  def setup
    @env_backup = {}
    ENV_KEYS.each do |key|
      @env_backup[key] = ENV.key?(key) ? ENV[key] : :__unset__
      ENV.delete(key)
    end
  end

  def teardown
    @env_backup.each do |key, value|
      if value == :__unset__
        ENV.delete(key)
      else
        ENV[key] = value
      end
    end
  end

  def test_auto_grad_accum_matches_target_batch
    ENV["NANOGPT_DEVICE"] = "cpu"
    ENV["NANOGPT_BATCH_SIZE"] = "16"
    ENV["NANOGPT_MICRO_BATCH_SIZE"] = "4"

    options = NanoGptShakespeareTrainer.resolve_options

    assert_equal 16, options.fetch(:batch_size)
    assert_equal 4, options.fetch(:micro_batch_size)
    assert_equal 4, options.fetch(:gradient_accumulation_steps)
    assert_equal 16, options.fetch(:effective_batch_size)
    assert_equal "auto", options.fetch(:grad_accum_mode)
    assert_equal [4, 4, 4, 4], options.fetch(:micro_batch_plan)
  end

  def test_manual_grad_accum_uses_fixed_micro_batches
    ENV["NANOGPT_BATCH_SIZE"] = "64"
    ENV["NANOGPT_MICRO_BATCH_SIZE"] = "8"
    ENV["NANOGPT_GRAD_ACCUM"] = "2"
    ENV["NANOGPT_UNSAFE_BATCHING"] = "1"

    options = NanoGptShakespeareTrainer.resolve_options

    assert_equal 2, options.fetch(:gradient_accumulation_steps)
    assert_equal 16, options.fetch(:effective_batch_size)
    assert_equal "manual", options.fetch(:grad_accum_mode)
    assert_equal [8, 8], options.fetch(:micro_batch_plan)
  end

  def test_micro_batch_size_is_capped_by_target_batch
    ENV["NANOGPT_BATCH_SIZE"] = "5"
    ENV["NANOGPT_MICRO_BATCH_SIZE"] = "32"
    ENV["NANOGPT_UNSAFE_BATCHING"] = "1"

    options = NanoGptShakespeareTrainer.resolve_options

    assert_equal 5, options.fetch(:micro_batch_size)
    assert_equal [5], options.fetch(:micro_batch_plan)
    assert_equal 5, options.fetch(:effective_batch_size)
  end

  def test_micro_batch_size_is_clamped_to_safe_default_without_unsafe_flag
    ENV["NANOGPT_BATCH_SIZE"] = "64"
    ENV["NANOGPT_MICRO_BATCH_SIZE"] = "64"

    options = NanoGptShakespeareTrainer.resolve_options

    assert_equal 1, options.fetch(:micro_batch_size)
    assert_equal 1, options.fetch(:eval_batch_size)
    assert_equal "auto", options.fetch(:grad_accum_mode)
  end

  def test_unsafe_batching_allows_requested_micro_batch_size
    ENV["NANOGPT_BATCH_SIZE"] = "64"
    ENV["NANOGPT_MICRO_BATCH_SIZE"] = "64"
    ENV["NANOGPT_UNSAFE_BATCHING"] = "1"

    options = NanoGptShakespeareTrainer.resolve_options

    assert_equal 64, options.fetch(:micro_batch_size)
    assert_equal [64], options.fetch(:micro_batch_plan)
  end

  def test_gradient_checkpointing_defaults_to_enabled
    options = NanoGptShakespeareTrainer.resolve_options
    assert_equal true, options.fetch(:gradient_checkpointing)
  end
end
