# frozen_string_literal: true

require "json"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase218DistributedConfigMainParityTest < Minitest::Test
  Host = MLX::DistributedUtils::Host
  SSHInfo = MLX::DistributedUtils::SSHInfo

  def setup
    TestSupport.build_native_extension!
    singleton = MLX::DistributedUtils.singleton_class
    singleton.class_eval do
      alias_method :_phase218_check_ssh_connections, :check_ssh_connections
      alias_method :_phase218_prepare_ethernet_hostfile, :prepare_ethernet_hostfile
      alias_method :_phase218_prepare_tb_hostfile, :prepare_tb_hostfile

      define_method(:check_ssh_connections) do |hosts, runner: nil|
        _ = runner
        @phase218_last_hosts = hosts
        hosts.map { SSHInfo.new(can_ssh: true, has_sudo: true) }
      end

      define_method(:prepare_ethernet_hostfile) do |args, hosts, runner: nil|
        _ = [runner]
        @phase218_last_call = [:ethernet, args, hosts]
      end

      define_method(:prepare_tb_hostfile) do |args, hosts, sshinfo, runner: nil|
        _ = [runner]
        @phase218_last_call = [:thunderbolt, args, hosts, sshinfo]
      end
    end
  end

  def teardown
    singleton = MLX::DistributedUtils.singleton_class
    singleton.class_eval do
      remove_method :check_ssh_connections
      remove_method :prepare_ethernet_hostfile
      remove_method :prepare_tb_hostfile
      alias_method :check_ssh_connections, :_phase218_check_ssh_connections
      alias_method :prepare_ethernet_hostfile, :_phase218_prepare_ethernet_hostfile
      alias_method :prepare_tb_hostfile, :_phase218_prepare_tb_hostfile
      remove_method :_phase218_check_ssh_connections
      remove_method :_phase218_prepare_ethernet_hostfile
      remove_method :_phase218_prepare_tb_hostfile
    end
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_main_dispatches_ethernet_path
    code = MLX::DistributedUtils.__send__(:config_main, ["--over", "ethernet", "--hosts", "h0,h1", "--env", "X=1"])
    assert_equal 0, code

    kind, args, hosts = MLX::DistributedUtils.instance_variable_get(:@phase218_last_call)
    assert_equal :ethernet, kind
    assert_equal ["X=1"], args[:env]
    assert_equal 2, hosts.length
    assert_equal "h0", hosts[0].ssh_hostname
    assert_equal "h1", hosts[1].ssh_hostname
  end

  def test_main_dispatches_thunderbolt_path
    Dir.mktmpdir("mlx-config-main") do |dir|
      hostfile = File.join(dir, "hosts.json")
      File.write(
        hostfile,
        JSON.dump(
          {
            "hosts" => [
              { "ssh" => "a", "ips" => ["10.0.0.1"], "rdma" => [] },
              { "ssh" => "b", "ips" => ["10.0.0.2"], "rdma" => [] }
            ]
          }
        )
      )

      code = MLX::DistributedUtils.__send__(:config_main, ["--over", "thunderbolt", "--hostfile", hostfile, "--backend", "ring"])
      assert_equal 0, code

      kind, args, hosts, sshinfo = MLX::DistributedUtils.instance_variable_get(:@phase218_last_call)
      assert_equal :thunderbolt, kind
      assert_equal "ring", args[:backend]
      assert_equal 2, hosts.length
      assert_equal 2, sshinfo.length
    end
  end
end
