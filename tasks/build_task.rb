# frozen_string_literal: true

require "rbconfig"

class BuildTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze
  CLEAN_PATTERNS = [
    "ext/mlx/build",
    "ext/mlx/Makefile",
    "ext/mlx/native.o",
    "ext/mlx/native.bundle",
    "ext/mlx/native.bundle.dSYM"
  ].freeze

  def self.clean_patterns
    CLEAN_PATTERNS
  end

  def self.build_native_extension!(make: ENV.fetch("MAKE", RbConfig::CONFIG["MAKE"] || "make"))
    ext_dir = File.join(REPO_ROOT, "ext", "mlx")
    run_command!([RbConfig.ruby, "extconf.rb"], chdir: ext_dir)
    run_command!([make], chdir: ext_dir)
  end

  def self.run_command!(command, chdir:)
    success = system(*command, chdir: chdir)
    return if success

    raise "command failed: #{command.join(' ')} (cwd: #{chdir})"
  end
end
