# frozen_string_literal: true

require "rbconfig"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase212DistributedRemoteProcessRuntimeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_remote_process_spawns_and_reports_exit_status
    proc = MLX::DistributedUtils::RemoteProcess.new(
      0,
      "127.0.0.1",
      RbConfig.ruby,
      nil,
      {},
      [],
      [RbConfig.ruby, "-e", "puts 'mlx-remote-ok'"]
    )

    refute_nil proc.process
    assert_respond_to proc.process, :poll

    wait_until(timeout: 5) { !proc.exit_status[0].nil? }
    status, killed = proc.exit_status
    assert_equal 0, status
    assert_equal false, killed
  end

  def test_remote_process_terminate_sets_killed_flag
    proc = MLX::DistributedUtils::RemoteProcess.new(
      0,
      "127.0.0.1",
      RbConfig.ruby,
      nil,
      {},
      [],
      [RbConfig.ruby, "-e", "sleep 10"]
    )

    wait_until(timeout: 2) { proc.process.respond_to?(:poll) && proc.process.poll.nil? }
    proc.terminate

    wait_until(timeout: 5) { !proc.exit_status[0].nil? }
    _status, killed = proc.exit_status
    assert_equal true, killed
  end

  private

  def wait_until(timeout:)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
    until yield
      raise "timeout waiting for condition" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline

      sleep 0.02
    end
  end
end
