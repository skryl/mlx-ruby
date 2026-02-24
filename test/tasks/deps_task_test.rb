# frozen_string_literal: true

require_relative "../support/test_helper"
require_relative "../../tasks/deps_task"

class DepsTaskTest < Minitest::Test
  def setup
    @singleton = class << DepsTask
      self
    end
    @restores = []
  end

  def teardown
    @restores.reverse_each do |name, backup|
      @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end
  end

  def test_install_all_forwards_python_bin_to_web_dependencies
    calls = []

    stub_singleton_method(:install_ruby_dependencies!) do
      calls << [:ruby, nil]
    end
    stub_singleton_method(:install_python_dependencies!) do |python_bin:|
      calls << [:python, python_bin]
    end
    stub_singleton_method(:install_web_dependencies!) do |python_bin:|
      calls << [:web, python_bin]
    end

    DepsTask.install_all!(python_bin: "python-custom")

    assert_equal [
      [:ruby, nil],
      [:python, "python-custom"],
      [:web, "python-custom"]
    ], calls
  end

  def test_install_web_dependencies_installs_and_verifies_web_smoke_requirements
    commands = []
    stub_singleton_method(:run_command!) do |command, chdir:|
      commands << [command, chdir]
    end

    DepsTask.install_web_dependencies!(python_bin: "python-custom")

    expected = [
      [%w[python-custom -m pip install onnx], DepsTask::REPO_ROOT],
      [%w[node --version], DepsTask::REPO_ROOT],
      [%w[npm --version], DepsTask::REPO_ROOT],
      [%w[npx --version], DepsTask::REPO_ROOT],
      [%w[npm ci], DepsTask::WEB_ROOT],
      [%w[npx playwright install chromium], DepsTask::WEB_ROOT],
      [["node", "-e", "import(process.argv[1]).then(() => process.exit(0)).catch(() => process.exit(1))", "playwright"], DepsTask::WEB_ROOT],
      [["node", "-e", "import(process.argv[1]).then(() => process.exit(0)).catch(() => process.exit(1))", "onnxruntime-web"], DepsTask::WEB_ROOT]
    ]

    assert_equal expected, commands
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__deps_task_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
    @singleton.define_method(name, &block)
  end
end
