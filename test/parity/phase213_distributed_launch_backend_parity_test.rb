# frozen_string_literal: true

require "json"
require "rbconfig"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase213DistributedLaunchBackendParityTest < Minitest::Test
  Host = MLX::DistributedUtils::Host

  def setup
    TestSupport.build_native_extension!
    @captured = nil
    MLX::DistributedUtils.singleton_class.class_eval do
      alias_method :_phase213_original_launch_with_io, :launch_with_io
      define_method(:launch_with_io) do |command_class, arguments, verbose|
        @_phase213_capture = {
          command_class: command_class,
          arguments: arguments,
          verbose: verbose
        }
        Array.new(arguments.length) { [0, false] }
      end
    end
  end

  def teardown
    MLX::DistributedUtils.singleton_class.class_eval do
      remove_method :launch_with_io
      alias_method :launch_with_io, :_phase213_original_launch_with_io
      remove_method :_phase213_original_launch_with_io
    end
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_launch_ring_builds_hostfile_and_env
    hosts = [
      Host.new(rank: 0, ssh_hostname: "127.0.0.1", ips: ["10.0.0.1"], rdma: []),
      Host.new(rank: 1, ssh_hostname: "worker", ips: ["10.0.0.2"], rdma: [])
    ]
    args = struct_with(
      starting_port: 32_000,
      connections_per_ip: 2,
      env: ["A=1"],
      verbose: true,
      cwd: "/tmp",
      python: RbConfig.ruby
    )
    command = [RbConfig.ruby, "-e", "puts :ok"]

    MLX::DistributedUtils.launch_ring(nil, hosts, args, command)
    capture = MLX::DistributedUtils.instance_variable_get(:@_phase213_capture)
    refute_nil capture
    assert_equal MLX::DistributedUtils::RemoteProcess, capture[:command_class]
    assert_equal true, capture[:verbose]
    assert_equal 2, capture[:arguments].length

    first_args = capture[:arguments].first.first
    assert_equal 0, first_args[0]
    assert_equal "127.0.0.1", first_args[1]
    assert_equal "/tmp", first_args[3]
    files = first_args[4]
    env = first_args[5]

    hostfile = JSON.parse(files.fetch("MLX_HOSTFILE"))
    assert_equal ["10.0.0.1:32000", "10.0.0.1:32001"], hostfile[0]
    assert_equal ["10.0.0.2:32002", "10.0.0.2:32003"], hostfile[1]
    assert_includes env, "A=1"
    assert_includes env, "MLX_RING_VERBOSE=1"
  end

  def test_launch_nccl_adds_world_and_rank_env
    hosts = [
      Host.new(rank: 0, ssh_hostname: "h0", ips: ["192.168.0.10"], rdma: []),
      Host.new(rank: 1, ssh_hostname: "h1", ips: ["192.168.0.11"], rdma: [])
    ]
    args = struct_with(
      nccl_port: 18_000,
      repeat_hosts: 2,
      env: ["X=1"],
      verbose: false,
      cwd: nil,
      python: RbConfig.ruby
    )
    MLX::DistributedUtils.launch_nccl(nil, hosts, args, [RbConfig.ruby, "-e", "puts :nccl"])

    capture = MLX::DistributedUtils.instance_variable_get(:@_phase213_capture)
    env_rank0 = capture[:arguments][0][0][5]
    env_rank1 = capture[:arguments][1][0][5]
    assert_includes env_rank0, "NCCL_HOST_IP=192.168.0.10"
    assert_includes env_rank0, "NCCL_PORT=18000"
    assert_includes env_rank0, "MLX_WORLD_SIZE=2"
    assert_includes env_rank0, "CUDA_VISIBLE_DEVICES=0"
    assert_includes env_rank1, "CUDA_VISIBLE_DEVICES=1"
  end

  def test_launch_jaccl_writes_rdma_file_and_ring_flag
    hosts = [
      Host.new(rank: 0, ssh_hostname: "h0", ips: ["10.0.0.1"], rdma: [nil, "rdma_en2"]),
      Host.new(rank: 1, ssh_hostname: "h1", ips: ["10.0.0.2"], rdma: ["rdma_en2", nil])
    ]
    args = struct_with(
      backend: "jaccl-ring",
      starting_port: 33_000,
      env: [],
      verbose: false,
      cwd: nil,
      python: RbConfig.ruby
    )
    MLX::DistributedUtils.launch_jaccl(nil, hosts, args, [RbConfig.ruby, "-e", "puts :jaccl"])

    capture = MLX::DistributedUtils.instance_variable_get(:@_phase213_capture)
    first_args = capture[:arguments].first.first
    files = first_args[4]
    env = first_args[5]

    assert_equal hosts.map(&:rdma), JSON.parse(files.fetch("MLX_IBV_DEVICES"))
    assert_includes env, "MLX_JACCL_COORDINATOR=10.0.0.1:33000"
    assert_includes env, "MLX_JACCL_RING=1"
  end

  def test_launch_ring_requires_ips
    parser = Object.new
    parser.define_singleton_method(:error) { |message| raise RuntimeError, message }
    hosts = [Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: [])]
    args = struct_with(starting_port: 1, connections_per_ip: 1, env: [], verbose: false, cwd: nil, python: RbConfig.ruby)

    err = assert_raises(RuntimeError) do
      MLX::DistributedUtils.launch_ring(parser, hosts, args, [RbConfig.ruby, "-e", "puts :x"])
    end
    assert_match(/requires IPs/, err.message)
  end

  private

  def struct_with(**kwargs)
    Struct.new(*kwargs.keys, keyword_init: true).new(**kwargs)
  end
end
