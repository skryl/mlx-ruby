# frozen_string_literal: true

require "json"
require "net/http"
require "open3"
require "socket"
require "timeout"
require_relative "../../support/test_helper"

module WebDemoIntegrationTestHelper
  WEB_DEMO_HOST = "127.0.0.1".freeze
  PROBE_SCRIPT = File.join(RUBY_ROOT, "test", "integration", "web", "demo_page_probe.mjs").freeze

  def run
    run_without_timeout
  end

  private

  def ensure_web_runtime_dependencies!
    skip "node is required for web integration tests" unless command_available?("node", "--version")
    skip "playwright module is required for web integration tests" unless node_module_available?("playwright")
    skip "onnxruntime-web module is required for web integration tests" unless node_module_available?("onnxruntime-web")
  end

  def ensure_demo_assets!(*relative_paths)
    missing = relative_paths.reject { |rel| File.exist?(File.join(RUBY_ROOT, rel)) }
    return if missing.empty?

    skip "missing web assets: #{missing.join(', ')} (run `bundle exec rake web:assets`)"
  end

  def with_web_demo_server
    host = WEB_DEMO_HOST
    port = reserve_free_port
    env = { "HOST" => host, "PORT" => port.to_s }
    stdin, stdout, stderr, wait_thr = Open3.popen3(
      env,
      "bundle",
      "exec",
      "rake",
      "web:serve",
      chdir: RUBY_ROOT
    )
    stdin.close
    wait_for_server!(host, port, wait_thr)
    yield("http://#{host}:#{port}")
  ensure
    stdout&.close
    stderr&.close
    shutdown_process!(wait_thr)
  end

  def probe_demo_page!(base_url, demo:)
    stdout, stderr, status = Open3.capture3(
      "node",
      PROBE_SCRIPT,
      "--base-url",
      base_url,
      "--demo",
      demo,
      chdir: File.join(RUBY_ROOT, "web")
    )

    unless status.success?
      raise <<~MSG
        demo probe failed for #{demo}
        stdout:
        #{stdout}
        stderr:
        #{stderr}
      MSG
    end

    JSON.parse(stdout)
  end

  def reserve_free_port
    server = TCPServer.new(WEB_DEMO_HOST, 0)
    port = server.addr[1]
    server.close
    port
  rescue Errno::EACCES, Errno::EPERM
    skip "web integration tests require binding to #{WEB_DEMO_HOST} sockets in this environment"
  end

  def wait_for_server!(host, port, wait_thr, timeout_seconds: 90)
    deadline = Time.now + timeout_seconds
    uri = URI("http://#{host}:#{port}/")

    loop do
      if wait_thr && !wait_thr.alive?
        raise "web server exited before becoming ready"
      end

      begin
        response = Net::HTTP.start(
          uri.host,
          uri.port,
          open_timeout: 1,
          read_timeout: 1
        ) { |http| http.get(uri.request_uri) }

        if response.is_a?(Net::HTTPSuccess) || response.is_a?(Net::HTTPRedirection)
          return
        end
      rescue Errno::ECONNREFUSED, Timeout::Error, Errno::EHOSTUNREACH
        nil
      end

      raise "timed out waiting for web server on #{uri}" if Time.now >= deadline

      sleep 0.25
    end
  end

  def shutdown_process!(wait_thr)
    return unless wait_thr

    pid = wait_thr.pid
    return unless pid
    return unless wait_thr.alive?

    begin
      Process.kill("TERM", pid)
    rescue Errno::ESRCH
      return
    end

    begin
      Timeout.timeout(5) { wait_thr.value }
    rescue Timeout::Error
      begin
        Process.kill("KILL", pid)
      rescue Errno::ESRCH
        nil
      end
      begin
        Timeout.timeout(5) { wait_thr.value }
      rescue Timeout::Error
        nil
      end
    end
  end

  def command_available?(*argv)
    _out, _err, status = Open3.capture3(*argv)
    status.success?
  rescue Errno::ENOENT
    false
  end

  def node_module_available?(name)
    _out, _err, status = Open3.capture3(
      "node",
      "-e",
      "import(process.argv[1]).then(() => process.exit(0)).catch(() => process.exit(1))",
      name,
      chdir: File.join(RUBY_ROOT, "web")
    )
    status.success?
  rescue Errno::ENOENT
    false
  end
end
