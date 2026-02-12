# frozen_string_literal: true

require "json"
require "minitest/autorun"
require "open3"
require "tmpdir"

RUBY_ROOT = File.expand_path("..", __dir__)
REPO_ROOT = File.expand_path("..", RUBY_ROOT)

module TestSupport
  module_function

  def build_native_extension!
    return if @native_built

    ext_dir = File.join(RUBY_ROOT, "ext", "mlx")
    run_cmd!(%w[ruby extconf.rb], ext_dir)
    run_cmd!(%w[make], ext_dir)
    @native_built = true
  end

  def run_cmd!(cmd, chdir)
    stdout, stderr, status = Open3.capture3(*cmd, chdir: chdir)
    return if status.success?

    raise <<~MSG
      command failed: #{cmd.join(" ")}
      cwd: #{chdir}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end
end
