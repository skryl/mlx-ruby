# frozen_string_literal: true

require "rbconfig"

class WebAssetsTask
  SCRIPTS_DIR = File.join(__dir__, "web_assets_task").freeze
  SCRIPT_NAMES = %w[
    export_gpt2_assets.rb
    export_stable_diffusion_assets.rb
    export_nanogpt_assets.rb
  ].freeze

  def self.script_paths
    SCRIPT_NAMES.map do |script_name|
      script_path = File.join(SCRIPTS_DIR, script_name)
      raise "Missing web assets export script: #{script_path}" unless File.exist?(script_path)

      script_path
    end
  end

  def self.run!(ruby_bin: RbConfig.ruby)
    script_paths.each do |script|
      success = system(ruby_bin, script)
      raise "Web assets export failed: #{script}" unless success
    end
  end
end
