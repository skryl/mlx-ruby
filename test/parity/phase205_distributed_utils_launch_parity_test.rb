# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase205DistributedUtilsLaunchParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_command_process_interface_contract
    process = MLX::DistributedUtils::CommandProcess.new
    assert_raises(NotImplementedError) { process.process }
    assert_raises(NotImplementedError) { process.exit_status }
    assert_raises(NotImplementedError) { process.terminate }
    assert_raises(NotImplementedError) { process.preprocess_output("x") }
  end

  def test_remote_process_launch_and_kill_script_generation
    script = MLX::DistributedUtils::RemoteProcess.make_launch_script(
      3,
      "/tmp",
      { "MLX_HOSTFILE" => "payload" },
      ["A=1"],
      ["python", "train.py"],
      true
    )
    assert_includes script, "MLX_RANK=3"
    assert_includes script, "export A=1"
    assert_includes script, "MLX_HOSTFILE"
    assert_includes script, "exec"

    kill_script = MLX::DistributedUtils::RemoteProcess.make_kill_script("/tmp/pidfile")
    assert_includes kill_script, "cat /tmp/pidfile"
    assert_includes kill_script, "kill $pid"
  end

  def test_get_mpi_libname_shape
    name = MLX::DistributedUtils.get_mpi_libname
    assert(name.nil? || name.is_a?(String))
  end
end
