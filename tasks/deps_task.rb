# frozen_string_literal: true

require_relative "benchmark_task"

class DepsTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze
  WEB_ROOT = File.join(REPO_ROOT, "web").freeze
  NODE_MODULE_CHECK_SCRIPT = "import(process.argv[1]).then(() => process.exit(0)).catch(() => process.exit(1))".freeze

  def self.install_all!(
    python_bin: ENV.fetch("PYTHON", "python3")
  )
    install_ruby_dependencies!
    install_python_dependencies!(python_bin: python_bin)
    install_web_dependencies!(python_bin: python_bin)
  end

  def self.install_ruby_dependencies!
    run_command!(%w[bundle install], chdir: REPO_ROOT)
  end

  def self.install_python_dependencies!(python_bin: ENV.fetch("PYTHON", "python3"))
    BenchmarkTask.install_dependencies!(python_bin: python_bin)
  end

  def self.install_web_dependencies!(python_bin: ENV.fetch("PYTHON", "python3"))
    package_json = File.join(WEB_ROOT, "package.json")
    return unless File.exist?(package_json)

    install_python_onnx_dependency!(python_bin: python_bin)
    verify_node_tooling!
    run_command!(%w[npm ci], chdir: WEB_ROOT)
    run_command!(%w[npx playwright install chromium], chdir: WEB_ROOT)
    verify_node_module_installed!("playwright")
    verify_node_module_installed!("onnxruntime-web")
  end

  def self.install_python_onnx_dependency!(python_bin:)
    commands = [
      [python_bin, "-m", "pip", "install", "onnx"],
      [python_bin, "-m", "pip", "install", "--break-system-packages", "onnx"],
      [python_bin, "-m", "pip", "install", "--user", "--break-system-packages", "onnx"]
    ]

    commands.each do |command|
      begin
        run_command!(command, chdir: REPO_ROOT)
        return
      rescue RuntimeError
        next
      end
    end

    raise "command failed: #{commands.last.join(' ')} (cwd: #{REPO_ROOT})"
  end

  def self.verify_node_tooling!
    run_command!(%w[node --version], chdir: REPO_ROOT)
    run_command!(%w[npm --version], chdir: REPO_ROOT)
    run_command!(%w[npx --version], chdir: REPO_ROOT)
  end

  def self.verify_node_module_installed!(name)
    run_command!(
      ["node", "-e", NODE_MODULE_CHECK_SCRIPT, name],
      chdir: WEB_ROOT
    )
  end

  def self.run_command!(command, chdir:)
    success = system(*command, chdir: chdir)
    return if success

    raise "command failed: #{command.join(' ')} (cwd: #{chdir})"
  end
end
