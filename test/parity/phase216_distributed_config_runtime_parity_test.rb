# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase216DistributedConfigRuntimeParityTest < Minitest::Test
  Host = MLX::DistributedUtils::Host
  TBPort = MLX::DistributedUtils::ThunderboltPort
  TBHost = MLX::DistributedUtils::ThunderboltHost

  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_ip_configurator_assigns_pairwise_ips
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
    refute_empty ips.ips[[0, 1]]
    refute_empty ips.ips[[1, 0]]
    assert_equal "en5", ips.ips[[0, 1]].first[0]
    assert_match(/\A192\.168\.\d+\.\d+\z/, ips.ips[[0, 1]].first[1])
  end

  def test_add_ips_and_check_rdma_use_runner
    hosts = [
      Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
      Host.new(rank: 1, ssh_hostname: "h1", ips: [], rdma: [])
    ]

    runner = lambda do |cmd|
      case cmd
      when ["ssh", "h0", "ipconfig", "getifaddr", "en0"]
        struct_with(stdout: "10.0.0.10\n", status: struct_with(success?: true))
      when ["ssh", "h1", "ipconfig", "getifaddr", "en0"]
        struct_with(stdout: "\n", status: struct_with(success?: true))
      when ["ssh", "h1", "ipconfig", "getifaddr", "en1"]
        struct_with(stdout: "10.0.0.11\n", status: struct_with(success?: true))
      when ["ssh", "h0", "ibv_devices"]
        struct_with(stdout: "device rdma_en2\n", status: struct_with(success?: true))
      when ["ssh", "h1", "ibv_devices"]
        struct_with(stdout: "no_devices\n", status: struct_with(success?: true))
      else
        struct_with(stdout: "", status: struct_with(success?: true))
      end
    end

    MLX::DistributedUtils.add_ips(hosts, verbose: false, runner: runner)
    assert_equal ["10.0.0.10"], hosts[0].ips
    assert_equal ["10.0.0.11"], hosts[1].ips

    _, _ = capture_io do
      refute MLX::DistributedUtils.check_rdma(hosts, verbose: false, strict: false, runner: runner)
      assert_raises(ArgumentError) do
        MLX::DistributedUtils.check_rdma(hosts, verbose: false, strict: true, runner: runner)
      end
    end
  end

  def test_check_ssh_connections_reports_sudo_flags
    hosts = [
      Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
      Host.new(rank: 1, ssh_hostname: "h1", ips: [], rdma: [])
    ]
    runner = lambda do |cmd|
      if cmd.include?("echo")
        ok = cmd.include?("h0")
        struct_with(stdout: "", status: struct_with(success?: ok, exitstatus: ok ? 0 : 1))
      elsif cmd.include?("sudo")
        struct_with(stdout: "", status: struct_with(success?: false, exitstatus: 1))
      else
        struct_with(stdout: "", status: struct_with(success?: true, exitstatus: 0))
      end
    end

    err = nil
    _, _ = capture_io do
      err = assert_raises(ArgumentError) do
        MLX::DistributedUtils.check_ssh_connections(hosts, runner: runner)
      end
    end
    assert_match(/Could not ssh/, err.message)
  end

  private

  def struct_with(**kwargs)
    Struct.new(*kwargs.keys, keyword_init: true).new(**kwargs)
  end
end
