# frozen_string_literal: true

require "fileutils"
require "json"
require "time"

NANOGPT_TRAIN_REPO_ROOT = File.expand_path("../..", __dir__)
NANOGPT_TRAIN_LIB_ROOT = File.join(NANOGPT_TRAIN_REPO_ROOT, "lib")
$LOAD_PATH.unshift(NANOGPT_TRAIN_LIB_ROOT) unless $LOAD_PATH.include?(NANOGPT_TRAIN_LIB_ROOT)

require "mlx"
require_relative "../../examples/web/nanogpt_example"

module NanoGptShakespeareTrainer
  module_function

  OUTPUT_DIR = File.join(NANOGPT_TRAIN_REPO_ROOT, "web", "assets", "nanogpt")
  WEIGHTS_PATH = File.join(OUTPUT_DIR, "weights.npz")
  TOKENIZER_PATH = File.join(OUTPUT_DIR, "tokenizer.json")
  TRAINING_CONFIG_PATH = File.join(OUTPUT_DIR, "training_config.json")
  TRAINING_SUMMARY_PATH = File.join(OUTPUT_DIR, "training_summary.json")

  PROMPT_PRESETS = {
    "to be" => "To be, or not to be",
    "romeo" => "Romeo, Romeo, wherefore art thou Romeo?",
    "king" => "KING HENRY:",
    "fools" => "The fool doth think he is wise,"
  }.freeze

  DEFAULTS = NanoGptExample::REFERENCE_CONFIG.merge(
    seed: 1337,
    device: "gpu",
    dataset_path: NanoGptExample::DEFAULT_DATASET_PATH
  ).freeze
  DEFAULT_GPU_MICRO_BATCH_SIZE = 1
  DEFAULT_CPU_MICRO_BATCH_SIZE = 4

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    options = resolve_options
    configure_device!(options.fetch(:device))
    MLX::Core.random_seed(options.fetch(:seed))

    text = NanoGptExample::Dataset.load_text(options.fetch(:dataset_path))
    tokenizer = NanoGptExample::Dataset.build_char_tokenizer(text)
    encoded = NanoGptExample::Dataset.encode(text, tokenizer)
    train_ids, val_ids = NanoGptExample::Dataset.split_train_val(encoded, train_fraction: 0.9)
    val_ids = train_ids if val_ids.nil? || val_ids.length <= options.fetch(:block_size) + 1

    model_config = {
      vocab_size: tokenizer.fetch("vocab_size"),
      block_size: options.fetch(:block_size),
      n_layer: options.fetch(:n_layer),
      n_head: options.fetch(:n_head),
      n_embd: options.fetch(:n_embd),
      dropout: options.fetch(:dropout),
      gradient_checkpointing: options.fetch(:gradient_checkpointing)
    }
    model = NanoGptExample::NanoGptModel.new(**model_config)
    model.train

    optimizer = MLX::Optimizers::AdamW.new(
      learning_rate: options.fetch(:learning_rate),
      betas: [options.fetch(:beta1), options.fetch(:beta2)],
      weight_decay: options.fetch(:weight_decay)
    )
    loss_and_grad = MLX::NN.value_and_grad(model, ->(inputs, targets) { NanoGptExample::Train.loss_fn(model, inputs, targets) })

    train_rng = Random.new(options.fetch(:seed) + 17)
    eval_rng = Random.new(options.fetch(:seed) + 29)

    puts "Training nanoGPT Shakespeare checkpoint for web demo"
    puts "  dataset: #{options.fetch(:dataset_path)} (chars=#{text.length}, vocab=#{tokenizer.fetch('vocab_size')})"
    puts "  model: layers=#{options.fetch(:n_layer)} heads=#{options.fetch(:n_head)} embd=#{options.fetch(:n_embd)} block=#{options.fetch(:block_size)} dropout=#{options.fetch(:dropout)}"
    puts "  memory: gradient_checkpointing=#{options.fetch(:gradient_checkpointing)}"
    puts "  training: steps=#{options.fetch(:max_iters)} target_batch=#{options.fetch(:batch_size)} lr=#{options.fetch(:learning_rate)} min_lr=#{options.fetch(:min_lr)} warmup=#{options.fetch(:warmup_iters)} decay_iters=#{options.fetch(:lr_decay_iters)}"
    puts "  batching: micro_batch=#{options.fetch(:micro_batch_size)} grad_accum=#{options.fetch(:gradient_accumulation_steps)} effective_batch=#{options.fetch(:effective_batch_size)} mode=#{options.fetch(:grad_accum_mode)}"
    puts "  eval: interval=#{options.fetch(:eval_interval)} iters=#{options.fetch(:eval_iters)} eval_batch=#{options.fetch(:eval_batch_size)}"
    puts "  device: #{options.fetch(:device)}"

    history = []
    final_train_loss = nil
    final_val_loss = nil

    options.fetch(:max_iters).times do |step_idx|
      lr = NanoGptExample::Train.learning_rate_for_step(
        step: step_idx,
        learning_rate: options.fetch(:learning_rate),
        min_lr: options.fetch(:min_lr),
        warmup_iters: options.fetch(:warmup_iters),
        lr_decay_iters: options.fetch(:lr_decay_iters)
      )
      optimizer.learning_rate = lr

      effective_batch = options.fetch(:effective_batch_size).to_f
      train_loss = 0.0
      accumulated_grads = nil

      options.fetch(:micro_batch_plan).each do |micro_batch_size|
        x, y = NanoGptExample::Dataset.sample_batch(
          token_ids: train_ids,
          batch_size: micro_batch_size,
          block_size: options.fetch(:block_size),
          rng: train_rng
        )

        loss, grads = loss_and_grad.call(x, y)
        weight = micro_batch_size.to_f / effective_batch
        train_loss += loss.item.to_f * weight
        weighted_grads = scale_gradient_tree(grads, weight)
        accumulated_grads = accumulated_grads.nil? ? weighted_grads : add_gradient_trees(accumulated_grads, weighted_grads)
        eval_gradient_tree(accumulated_grads)
      end

      optimizer.update(model, accumulated_grads)
      MLX::Core.eval(model.parameters, optimizer.state)
      final_train_loss = train_loss

      step_number = step_idx + 1
      row = {
        "step" => step_number,
        "learning_rate" => lr,
        "train_loss" => train_loss
      }

      should_eval = (step_number % options.fetch(:eval_interval)).zero? || step_number == options.fetch(:max_iters)
      if should_eval
        model.eval
        val_loss = evaluate(
          model: model,
          token_ids: val_ids,
          batch_size: options.fetch(:eval_batch_size),
          block_size: options.fetch(:block_size),
          eval_iters: options.fetch(:eval_iters),
          rng: eval_rng
        )
        model.train

        row["val_loss"] = val_loss
        final_val_loss = val_loss
        puts format("step=%5d lr=%.6f train_loss=%.5f val_loss=%.5f", step_number, lr, train_loss, val_loss)
      elsif (step_number % options.fetch(:log_interval)).zero?
        puts format("step=%5d lr=%.6f train_loss=%.5f", step_number, lr, train_loss)
      end
      history << row
    end

    model.eval
    preview = NanoGptExample::Train.sample_text(
      model: model,
      tokenizer: tokenizer,
      prompt: PROMPT_PRESETS.fetch("to be"),
      block_size: options.fetch(:block_size),
      pad_id: tokenizer.fetch("pad_id"),
      max_new_tokens: 200,
      temperature: 0.8,
      rng: Random.new(options.fetch(:seed) + 53)
    )

    FileUtils.mkdir_p(OUTPUT_DIR)
    model.save_weights(WEIGHTS_PATH)
    File.binwrite(TOKENIZER_PATH, JSON.pretty_generate(tokenizer))

    training_config = {
      "format" => "nanogpt_shakespeare_training_config_v1",
      "trained_at" => Time.now.utc.iso8601,
      "seed" => options.fetch(:seed),
      "source_reference" => "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
      "dataset" => "shakespeare_char",
      "dataset_path" => options.fetch(:dataset_path),
      "model" => stringify_keys(model_config),
      "training" => {
        "max_iters" => options.fetch(:max_iters),
        "batch_size" => options.fetch(:batch_size),
        "micro_batch_size" => options.fetch(:micro_batch_size),
        "gradient_accumulation_steps" => options.fetch(:gradient_accumulation_steps),
        "effective_batch_size" => options.fetch(:effective_batch_size),
        "grad_accum_mode" => options.fetch(:grad_accum_mode),
        "learning_rate" => options.fetch(:learning_rate),
        "min_lr" => options.fetch(:min_lr),
        "lr_decay_iters" => options.fetch(:lr_decay_iters),
        "warmup_iters" => options.fetch(:warmup_iters),
        "beta1" => options.fetch(:beta1),
        "beta2" => options.fetch(:beta2),
        "weight_decay" => options.fetch(:weight_decay),
        "eval_interval" => options.fetch(:eval_interval),
        "eval_iters" => options.fetch(:eval_iters),
        "eval_batch_size" => options.fetch(:eval_batch_size),
        "gradient_checkpointing" => options.fetch(:gradient_checkpointing),
        "log_interval" => options.fetch(:log_interval),
        "device" => options.fetch(:device)
      },
      "prompt_presets" => PROMPT_PRESETS
    }
    File.binwrite(TRAINING_CONFIG_PATH, JSON.pretty_generate(training_config))

    summary = {
      "format" => "nanogpt_shakespeare_training_summary_v1",
      "trained_at" => Time.now.utc.iso8601,
      "final_train_loss" => final_train_loss,
      "final_val_loss" => final_val_loss,
      "history" => history,
      "preview" => {
        "prompt" => PROMPT_PRESETS.fetch("to be"),
        "text" => preview
      }
    }
    File.binwrite(TRAINING_SUMMARY_PATH, JSON.pretty_generate(summary))

    puts "Saved nanoGPT training artifacts:"
    puts "  - #{WEIGHTS_PATH}"
    puts "  - #{TOKENIZER_PATH}"
    puts "  - #{TRAINING_CONFIG_PATH}"
    puts "  - #{TRAINING_SUMMARY_PATH}"
    puts "Preview:"
    puts preview
  end

  def evaluate(model:, token_ids:, batch_size:, block_size:, eval_iters:, rng:)
    losses = Array.new(eval_iters) do
      x, y = NanoGptExample::Dataset.sample_batch(
        token_ids: token_ids,
        batch_size: batch_size,
        block_size: block_size,
        rng: rng
      )
      loss = NanoGptExample::Train.loss_fn(model, x, y)
      MLX::Core.eval(loss)
      loss.item.to_f
    end
    losses.sum / losses.length.to_f
  end

  def stringify_keys(hash)
    hash.each_with_object({}) { |(key, value), out| out[key.to_s] = value }
  end

  def resolve_options
    device = ENV.fetch("NANOGPT_DEVICE", DEFAULTS.fetch(:device)).to_s.strip.downcase
    target_batch_size = env_int("NANOGPT_BATCH_SIZE", DEFAULTS.fetch(:batch_size), min: 1)
    micro_batch_default = default_micro_batch_size(device: device, batch_size: target_batch_size)
    grad_accum_override = env_optional_int("NANOGPT_GRAD_ACCUM", min: 1)
    unsafe_batching = ENV.fetch("NANOGPT_UNSAFE_BATCHING", "").to_s.strip == "1"

    normalize_batching_options({
      seed: env_int("NANOGPT_SEED", DEFAULTS.fetch(:seed), min: 0),
      dataset_path: env_path("NANOGPT_DATASET", DEFAULTS.fetch(:dataset_path)),
      device: device,
      max_iters: env_int("NANOGPT_STEPS", DEFAULTS.fetch(:max_iters), min: 1),
      batch_size: target_batch_size,
      micro_batch_size: env_int("NANOGPT_MICRO_BATCH_SIZE", micro_batch_default, min: 1),
      block_size: env_int("NANOGPT_BLOCK_SIZE", DEFAULTS.fetch(:block_size), min: 8),
      gradient_accumulation_steps: grad_accum_override,
      n_layer: env_int("NANOGPT_N_LAYER", DEFAULTS.fetch(:n_layer), min: 1),
      n_head: env_int("NANOGPT_N_HEAD", DEFAULTS.fetch(:n_head), min: 1),
      n_embd: env_int("NANOGPT_N_EMBD", DEFAULTS.fetch(:n_embd), min: 8),
      dropout: env_float("NANOGPT_DROPOUT", DEFAULTS.fetch(:dropout), min: 0.0),
      learning_rate: env_float("NANOGPT_LR", DEFAULTS.fetch(:learning_rate), min: 1e-8),
      min_lr: env_float("NANOGPT_MIN_LR", DEFAULTS.fetch(:min_lr), min: 1e-9),
      lr_decay_iters: env_int("NANOGPT_LR_DECAY_ITERS", DEFAULTS.fetch(:lr_decay_iters), min: 1),
      warmup_iters: env_int("NANOGPT_WARMUP_ITERS", DEFAULTS.fetch(:warmup_iters), min: 1),
      beta1: env_float("NANOGPT_BETA1", DEFAULTS.fetch(:beta1), min: 0.0),
      beta2: env_float("NANOGPT_BETA2", DEFAULTS.fetch(:beta2), min: 0.0),
      weight_decay: env_float("NANOGPT_WEIGHT_DECAY", DEFAULTS.fetch(:weight_decay), min: 0.0),
      eval_interval: env_int("NANOGPT_EVAL_INTERVAL", DEFAULTS.fetch(:eval_interval), min: 1),
      eval_iters: env_int("NANOGPT_EVAL_ITERS", DEFAULTS.fetch(:eval_iters), min: 1),
      eval_batch_size: env_int("NANOGPT_EVAL_BATCH_SIZE", micro_batch_default, min: 1),
      log_interval: env_int("NANOGPT_LOG_INTERVAL", DEFAULTS.fetch(:log_interval), min: 1),
      gradient_checkpointing: env_bool("NANOGPT_GRAD_CHECKPOINT", true),
      unsafe_batching: unsafe_batching
    })
  end

  def normalize_batching_options(options)
    target_batch_size = options.fetch(:batch_size)
    requested_micro_batch_size = options.fetch(:micro_batch_size)
    safe_cap = options.fetch(:unsafe_batching) ? target_batch_size : default_micro_batch_size(
      device: options.fetch(:device),
      batch_size: target_batch_size
    )
    micro_batch_size = [requested_micro_batch_size, target_batch_size, safe_cap].min
    grad_accum_override = options.fetch(:gradient_accumulation_steps)

    if requested_micro_batch_size > micro_batch_size
      warn "Clamping NANOGPT_MICRO_BATCH_SIZE from #{requested_micro_batch_size} to #{micro_batch_size} (set NANOGPT_UNSAFE_BATCHING=1 to bypass)"
    end

    if grad_accum_override.nil?
      micro_batch_plan = build_micro_batch_plan(total_samples: target_batch_size, micro_batch_size: micro_batch_size)
      grad_accum_steps = micro_batch_plan.length
      effective_batch_size = target_batch_size
      grad_accum_mode = "auto"
    else
      grad_accum_steps = grad_accum_override
      micro_batch_plan = Array.new(grad_accum_steps, micro_batch_size)
      effective_batch_size = micro_batch_size * grad_accum_steps
      grad_accum_mode = "manual"
    end

    options.merge(
      micro_batch_size: micro_batch_size,
      gradient_accumulation_steps: grad_accum_steps,
      effective_batch_size: effective_batch_size,
      grad_accum_mode: grad_accum_mode,
      micro_batch_plan: micro_batch_plan,
      eval_batch_size: [options.fetch(:eval_batch_size), micro_batch_size].min
    )
  end

  def build_micro_batch_plan(total_samples:, micro_batch_size:)
    plan = []
    remaining = Integer(total_samples)
    micro_batch_size = Integer(micro_batch_size)
    while remaining.positive?
      step_size = [micro_batch_size, remaining].min
      plan << step_size
      remaining -= step_size
    end
    plan
  end

  def default_micro_batch_size(device:, batch_size:)
    suggestion = if device == "cpu"
      DEFAULT_CPU_MICRO_BATCH_SIZE
    else
      DEFAULT_GPU_MICRO_BATCH_SIZE
    end
    [suggestion, batch_size].min
  end

  def scale_gradient_tree(grads, scale)
    MLX::Utils.tree_map(lambda { |grad| MLX::Core.multiply(grad, scale) }, grads)
  end

  def add_gradient_trees(left, right)
    MLX::Utils.tree_map(lambda { |lhs, rhs| MLX::Core.add(lhs, rhs) }, left, right)
  end

  def eval_gradient_tree(tree)
    flat = MLX::Utils.tree_flatten(tree)
    arrays = flat.map { |_path, value| value }
    return if arrays.empty?

    MLX::Core.eval(*arrays)
  end

  def env_path(name, default)
    value = ENV.fetch(name, "").to_s.strip
    return default if value.empty?

    value
  end

  def env_int(name, default, min: nil)
    value = ENV.fetch(name, "").to_s.strip
    return default if value.empty?

    parsed = Integer(value, 10)
    if !min.nil? && parsed < min
      abort("#{name} must be >= #{min}, got #{parsed}")
    end
    parsed
  rescue ArgumentError
    abort("#{name} must be an integer, got #{value.inspect}")
  end

  def env_optional_int(name, min: nil)
    value = ENV.fetch(name, "").to_s.strip
    return nil if value.empty?

    parsed = Integer(value, 10)
    if !min.nil? && parsed < min
      abort("#{name} must be >= #{min}, got #{parsed}")
    end
    parsed
  rescue ArgumentError
    abort("#{name} must be an integer, got #{value.inspect}")
  end

  def env_float(name, default, min: nil)
    value = ENV.fetch(name, "").to_s.strip
    return default if value.empty?

    parsed = Float(value)
    if !min.nil? && parsed < min
      abort("#{name} must be >= #{min}, got #{parsed}")
    end
    parsed
  rescue ArgumentError
    abort("#{name} must be a float, got #{value.inspect}")
  end

  def env_bool(name, default)
    value = ENV.fetch(name, "").to_s.strip.downcase
    return default if value.empty?

    return true if %w[1 true yes on].include?(value)
    return false if %w[0 false no off].include?(value)

    abort("#{name} must be a boolean (1/0 true/false), got #{value.inspect}")
  end

  def configure_device!(mode)
    case mode
    when "", "gpu", "default"
      nil
    when "cpu"
      MLX::Core.set_default_device(MLX::Core.cpu)
    else
      abort("Unsupported NANOGPT_DEVICE=#{mode.inspect}; expected cpu or gpu")
    end
  end
end

if $PROGRAM_NAME == __FILE__
  NanoGptShakespeareTrainer.run!
end
