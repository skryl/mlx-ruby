# frozen_string_literal: true

require "fileutils"
require "rbconfig"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase214DistributedLaunchMpiParityTest < Minitest::Test
  Host = MLX::DistributedUtils::Host

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_launch_mpi_builds_expected_command
    Dir.mktmpdir("mlx-mpi") do |dir|
      capture_file = File.join(dir, "captured_args.txt")
      mpirun = File.join(dir, "mpirun")
      File.write(
        mpirun,
        <<~SH
          #!/bin/bash
          : "${MLX_MPI_CAPTURE:?missing MLX_MPI_CAPTURE}"
          printf "%s\\n" "$@" > "$MLX_MPI_CAPTURE"
          exit 0
        SH
      )
      FileUtils.chmod("+x", mpirun)

      old_path = ENV["PATH"]
      old_capture = ENV["MLX_MPI_CAPTURE"]
      ENV["PATH"] = "#{dir}:#{old_path}"
      ENV["MLX_MPI_CAPTURE"] = capture_file

      hosts = [
        Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
        Host.new(rank: 1, ssh_hostname: "h0", ips: [], rdma: []),
        Host.new(rank: 2, ssh_hostname: "h1", ips: [], rdma: [])
      ]
      args = struct_with(
        env: ["A=1"],
        cwd: "/tmp",
        mpi_arg: ["--bind-to none"],
        verbose: false
      )
      command = [RbConfig.ruby, "-e", "puts :mpi"]

      status = MLX::DistributedUtils.launch_mpi(nil, hosts, args, command)
      assert_equal 0, status

      tokens = File.read(capture_file).split("\n")
      assert_includes tokens, "--output"
      assert_includes tokens, ":raw"
      assert_includes tokens, "--hostfile"
      assert_includes tokens, "-cwd"
      assert_includes tokens, "/tmp"
      assert_includes tokens, "-x"
      assert_includes tokens, "A=1"
      assert_includes tokens, "--bind-to"
      assert_includes tokens, "none"
      assert_includes tokens, "--"
      assert_includes tokens, RbConfig.ruby
      assert_includes tokens, "-e"
      assert_includes tokens, "puts :mpi"
    ensure
      ENV["PATH"] = old_path
      ENV["MLX_MPI_CAPTURE"] = old_capture
    end
  end

  def test_get_mpi_libname_returns_nil_or_string
    out = MLX::DistributedUtils.get_mpi_libname
    assert(out.nil? || out.is_a?(String))
  end

  private

  def struct_with(**kwargs)
    Struct.new(*kwargs.keys, keyword_init: true).new(**kwargs)
  end
end
