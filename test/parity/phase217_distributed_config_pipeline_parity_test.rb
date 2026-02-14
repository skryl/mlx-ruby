# frozen_string_literal: true

require "json"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase217DistributedConfigPipelineParityTest < Minitest::Test
  Host = MLX::DistributedUtils::Host
  TBPort = MLX::DistributedUtils::ThunderboltPort
  TBHost = MLX::DistributedUtils::ThunderboltHost
  SSHInfo = MLX::DistributedUtils::SSHInfo

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_configure_ring_jaccl_and_jaccl_ring_emit_hostfiles
    Dir.mktmpdir("mlx-config-217") do |dir|
      hosts = [
        Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
        Host.new(rank: 1, ssh_hostname: "h1", ips: [], rdma: [])
      ]
      tb_hosts = [
        TBHost.new(name: "h0", ports: [TBPort.new(iface: "en5", uuid: "u1", connected_to: "u2")]),
        TBHost.new(name: "h1", ports: [TBPort.new(iface: "en6", uuid: "u2", connected_to: "u1")])
      ]
      reverse = { "u1" => [0, 0], "u2" => [1, 0] }
      ips = MLX::DistributedUtils::IPConfigurator.new(hosts, tb_hosts, reverse)
      sshinfo = [SSHInfo.new(can_ssh: true, has_sudo: true), SSHInfo.new(can_ssh: true, has_sudo: true)]

      runner = lambda do |cmd|
        case cmd
        when ["ssh", "h0", "ipconfig", "getifaddr", "en0"]
          struct_with(stdout: "10.0.0.10\n", status: struct_with(success?: true))
        when ["ssh", "h1", "ipconfig", "getifaddr", "en0"]
          struct_with(stdout: "10.0.0.11\n", status: struct_with(success?: true))
        else
          struct_with(stdout: "", status: struct_with(success?: true))
        end
      end

      ring_file = File.join(dir, "ring.json")
      ring_args = struct_with(output_hostfile: ring_file, env: ["E=1"], verbose: false, auto_setup: true)
      MLX::DistributedUtils.configure_ring(ring_args, hosts, ips, [[0, 1], 1], sshinfo, runner: runner)
      ring_payload = JSON.parse(File.read(ring_file))
      assert_equal "ring", ring_payload["backend"]
      assert_equal ["E=1"], ring_payload["envs"]
      assert_equal 2, ring_payload["hosts"].length

      jaccl_file = File.join(dir, "jaccl.json")
      jaccl_args = struct_with(output_hostfile: jaccl_file, env: ["E=2"], verbose: false, auto_setup: true)
      MLX::DistributedUtils.configure_jaccl(jaccl_args, hosts, ips, sshinfo, runner: runner)
      jaccl_payload = JSON.parse(File.read(jaccl_file))
      assert_equal "jaccl", jaccl_payload["backend"]
      assert_equal 2, jaccl_payload["hosts"].length
      assert_equal [nil, "rdma_en5"], jaccl_payload["hosts"][0]["rdma"]
      assert_equal ["rdma_en6", nil], jaccl_payload["hosts"][1]["rdma"]

      jaccl_ring_file = File.join(dir, "jaccl-ring.json")
      jaccl_ring_args = struct_with(output_hostfile: jaccl_ring_file, env: ["E=3"], verbose: false, auto_setup: true)
      MLX::DistributedUtils.configure_jaccl_ring(jaccl_ring_args, hosts, ips, [[0, 1], 1], sshinfo, runner: runner)
      jaccl_ring_payload = JSON.parse(File.read(jaccl_ring_file))
      assert_equal "jaccl-ring", jaccl_ring_payload["backend"]
      assert_equal [nil, "rdma_en5"], jaccl_ring_payload["hosts"][0]["rdma"]
      assert_equal ["rdma_en6", nil], jaccl_ring_payload["hosts"][1]["rdma"]
    end
  end

  def test_prepare_ethernet_hostfile_uses_add_ips
    Dir.mktmpdir("mlx-eth-217") do |dir|
      hosts = [
        Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
        Host.new(rank: 1, ssh_hostname: "h1", ips: [], rdma: [])
      ]
      runner = lambda do |cmd|
        case cmd
        when ["ssh", "h0", "ipconfig", "getifaddr", "en0"]
          struct_with(stdout: "10.1.0.10\n", status: struct_with(success?: true))
        when ["ssh", "h1", "ipconfig", "getifaddr", "en0"]
          struct_with(stdout: "10.1.0.11\n", status: struct_with(success?: true))
        else
          struct_with(stdout: "", status: struct_with(success?: true))
        end
      end

      out_file = File.join(dir, "ethernet.json")
      args = struct_with(output_hostfile: out_file, env: ["ENV_X=1"], verbose: false)
      MLX::DistributedUtils.prepare_ethernet_hostfile(args, hosts, runner: runner)

      payload = JSON.parse(File.read(out_file))
      assert_equal "", payload["backend"]
      assert_equal ["ENV_X=1"], payload["envs"]
      assert_equal ["10.1.0.10"], payload["hosts"][0]["ips"]
      assert_equal ["10.1.0.11"], payload["hosts"][1]["ips"]
    end
  end

  private

  def struct_with(**kwargs)
    Struct.new(*kwargs.keys, keyword_init: true).new(**kwargs)
  end
end
