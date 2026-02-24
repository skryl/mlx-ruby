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

  def self.run!(ruby_bin: RbConfig.ruby, out: $stdout)
    out.puts "[web:assets] Resolving export scripts..."
    scripts = script_paths
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

  def self.monotonic_now
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end
end
