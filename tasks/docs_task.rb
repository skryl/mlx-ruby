# frozen_string_literal: true

require "rbconfig"

class DocsTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze

  def self.build!(make: ENV.fetch("MAKE", RbConfig::CONFIG["MAKE"] || "make"))
    docs_dir = File.join(REPO_ROOT, "docs")
    run_command!(["doxygen"], chdir: docs_dir)
    run_command!([make, "html"], chdir: docs_dir)
  end

  def self.run_command!(command, chdir:)
    success = system(*command, chdir: chdir)
    return if success

    raise "command failed: #{command.join(' ')} (cwd: #{chdir})"
  end
end
