# frozen_string_literal: true

require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase204DistributedUtilsConfigParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_parse_hardware_ports
    raw = <<~TXT
      Hardware Port: Thunderbolt 1
      Device: en5
      Hardware Port: Wi-Fi
      Device: en0
    TXT

    ports = MLX::DistributedUtils.parse_hardware_ports(raw)
    assert_equal "en5", ports["Thunderbolt 1"]
    assert_equal "en0", ports["Wi-Fi"]
  end

  def test_connectivity_matrix_and_ring_extraction
    p0 = MLX::DistributedUtils::ThunderboltPort.new(iface: "en5", uuid: "u1", connected_to: "u2")
    p1 = MLX::DistributedUtils::ThunderboltPort.new(iface: "en6", uuid: "u2", connected_to: "u1")
    h0 = MLX::DistributedUtils::ThunderboltHost.new(name: "h0", ports: [p0])
    h1 = MLX::DistributedUtils::ThunderboltHost.new(name: "h1", ports: [p1])

    reverse = { "u1" => [0, 0], "u2" => [1, 0] }
    connectivity = MLX::DistributedUtils.make_connectivity_matrix([h0, h1], reverse)
    assert_equal [[0, 1], [1, 0]], connectivity

    rings = MLX::DistributedUtils.extract_rings(connectivity)
    refute_empty rings
    assert_equal [0, 1], rings.first[0]
    assert_equal 1, rings.first[1]
  end

  def test_mesh_ring_validation_and_auto_setup
    hosts = [
      MLX::DistributedUtils::Host.new(rank: 0, ssh_hostname: "h0", ips: [], rdma: []),
      MLX::DistributedUtils::Host.new(rank: 1, ssh_hostname: "h1", ips: [], rdma: [])
    ]

    assert MLX::DistributedUtils.check_valid_mesh(hosts, [[0, 1], [1, 0]], strict: false)
    refute MLX::DistributedUtils.check_valid_mesh(hosts, [[0, 0], [0, 0]], strict: false)
    assert_raises(ArgumentError) { MLX::DistributedUtils.check_valid_mesh(hosts, [[0, 0], [0, 0]], strict: true) }

    assert MLX::DistributedUtils.check_valid_ring(hosts, [[[0, 1], 1]], strict: false)
    refute MLX::DistributedUtils.check_valid_ring(hosts, [], strict: false)
    assert_raises(ArgumentError) { MLX::DistributedUtils.check_valid_ring(hosts, [], strict: true) }

    sshinfo = [
      MLX::DistributedUtils::SSHInfo.new(can_ssh: true, has_sudo: true),
      MLX::DistributedUtils::SSHInfo.new(can_ssh: true, has_sudo: false)
    ]
    refute MLX::DistributedUtils.can_auto_setup(hosts, sshinfo, auto_setup: true)
  end
end
