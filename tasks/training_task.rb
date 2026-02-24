# frozen_string_literal: true

require "rbconfig"

class TrainingTask
  SCRIPTS_DIR = File.join(__dir__, "training_task").freeze
  DEFAULT_MODEL = "nanogpt"
  SCRIPT_BY_MODEL = {
    "nanogpt" => "train_nanogpt_shakespeare.rb"
  }.freeze

  def self.supported_models
    SCRIPT_BY_MODEL.keys
  end

  def self.script_for(model)
    normalized = normalize_model(model)
    script_name = SCRIPT_BY_MODEL[normalized]
    if script_name.nil?
      raise "Unknown web training model '#{normalized}'. Supported models: #{supported_models.join(', ')}."
    end

    script_path = File.join(SCRIPTS_DIR, script_name)
    raise "Missing training script: #{script_path}" unless File.exist?(script_path)

    script_path
  end

  def self.normalize_model(model)
    value = model.to_s.strip.downcase
    return DEFAULT_MODEL if value.empty?

    value
  end

  def self.run!(model = nil, ruby_bin: RbConfig.ruby)
    script = script_for(model)
    success = system(ruby_bin, script)
    return if success

    raise "Training script failed: #{script}"
  end
end
