# frozen_string_literal: true

require "rbconfig"

class WebAssetsTask
  SCRIPTS_DIR = File.join(__dir__, "web_assets_task").freeze
  SCRIPT_NAMES_BY_TARGET = {
    "gpt2" => "export_gpt2_assets.rb",
    "stable_diffusion" => "export_stable_diffusion_assets.rb",
    "nanogpt" => "export_nanogpt_assets.rb"
  }.freeze

  def self.script_paths(targets: selected_targets)
    targets.map do |target|
      script_name = SCRIPT_NAMES_BY_TARGET.fetch(target)
      script_path = File.join(SCRIPTS_DIR, script_name)
      raise "Missing web assets export script: #{script_path}" unless File.exist?(script_path)

      script_path
    end
  end

  def self.run!(ruby_bin: RbConfig.ruby, out: $stdout)
    targets = selected_targets
    out.puts "[web:assets] Resolving export scripts..."
    scripts = script_paths(targets: targets)
    out.puts "[web:assets] Targets: #{targets.join(', ')}"
    out.puts "[web:assets] Found #{scripts.length} script(s) to run."
    scripts.each_with_index do |script, index|
      out.puts "[web:assets]   #{index + 1}. #{script}"
    end
    out.puts "[web:assets] Ruby executable: #{ruby_bin}"

    started_at = monotonic_now
    scripts.each_with_index do |script, index|
      step_label = "#{index + 1}/#{scripts.length}"
      script_name = File.basename(script)
      script_started_at = monotonic_now

      out.puts "[web:assets] (#{step_label}) Starting #{script_name}"
      success = system(ruby_bin, script)
      elapsed = monotonic_now - script_started_at

      if success
        out.puts format(
          "[web:assets] (#{step_label}) Completed %<script>s in %<elapsed>.2fs",
          script: script_name,
          elapsed: elapsed
        )
      else
        out.puts format(
          "[web:assets] (#{step_label}) Failed %<script>s after %<elapsed>.2fs",
          script: script_name,
          elapsed: elapsed
        )
        raise "Web assets export failed: #{script}"
      end
    end

    total_elapsed = monotonic_now - started_at
    out.puts format("[web:assets] Finished web asset export in %.2fs", total_elapsed)
  end

  def self.selected_targets(raw_targets = ENV["WEB_ASSETS_TARGETS"])
    available = SCRIPT_NAMES_BY_TARGET.keys
    return available if raw_targets.nil? || raw_targets.strip.empty?

    requested = raw_targets.split(",").map(&:strip).reject(&:empty?)
    unknown = requested - available
    unless unknown.empty?
      raise ArgumentError,
            "Unknown WEB_ASSETS_TARGETS values: #{unknown.join(', ')} (supported: #{available.join(', ')})"
    end

    available.select { |target| requested.include?(target) }
  end

  def self.monotonic_now
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end
end
