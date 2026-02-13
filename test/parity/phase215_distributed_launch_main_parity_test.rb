# frozen_string_literal: true

require "json"
require "tmpdir"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase215DistributedLaunchMainParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    @captures = []
    singleton = MLX::DistributedUtils.singleton_class
    singleton.class_eval do
      alias_method :_phase215_launch_ring, :launch_ring
      alias_method :_phase215_launch_nccl, :launch_nccl
      alias_method :_phase215_launch_jaccl, :launch_jaccl
      alias_method :_phase215_launch_mpi, :launch_mpi

      define_method(:launch_ring) do |parser, hosts, args, command|
        @captures ||= []
        @captures << [:ring, parser, hosts, args, command]
        0
      end

      define_method(:launch_nccl) do |parser, hosts, args, command|
        @captures ||= []
        @captures << [:nccl, parser, hosts, args, command]
        0
      end

      define_method(:launch_jaccl) do |parser, hosts, args, command|
        @captures ||= []
        @captures << [:jaccl, parser, hosts, args, command]
        0
      end

      define_method(:launch_mpi) do |parser, hosts, args, command|
        @captures ||= []
        @captures << [:mpi, parser, hosts, args, command]
        0
      end
    end
  end

  def teardown
    singleton = MLX::DistributedUtils.singleton_class
    singleton.class_eval do
      remove_method :launch_ring
      remove_method :launch_nccl
      remove_method :launch_jaccl
      remove_method :launch_mpi
      alias_method :launch_ring, :_phase215_launch_ring
      alias_method :launch_nccl, :_phase215_launch_nccl
      alias_method :launch_jaccl, :_phase215_launch_jaccl
      alias_method :launch_mpi, :_phase215_launch_mpi
      remove_method :_phase215_launch_ring
      remove_method :_phase215_launch_nccl
      remove_method :_phase215_launch_jaccl
      remove_method :_phase215_launch_mpi
    end
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_main_dispatches_backend_and_resolves_command
    code = MLX::DistributedUtils.main(["--backend", "ring", "ruby", "-e", "puts :ok"])
    assert_equal 0, code

    capture = MLX::DistributedUtils.instance_variable_get(:@captures).last
    refute_nil capture
    kind, _parser, hosts, args, command = capture
    assert_equal :ring, kind
    assert_equal 1, hosts.length
    assert_equal "127.0.0.1", hosts.first.ssh_hostname
    assert_equal "ring", args[:backend]
    assert_equal "ruby", File.basename(command.first)
    assert_equal ["-e", "puts :ok"], command[1..]
  end

  def test_main_uses_hostfile_backend_and_env
    Dir.mktmpdir("mlx-launch-main") do |dir|
      hostfile = File.join(dir, "hosts.json")
      File.write(
        hostfile,
        JSON.dump(
          {
            "backend" => "jaccl",
            "envs" => ["HOSTFILE_ENV=1"],
            "hosts" => [{ "ssh" => "h0", "ips" => ["10.0.0.1"], "rdma" => [nil] }]
          }
        )
      )

      code = MLX::DistributedUtils.main(["--hostfile", hostfile, "--python", "ruby", "ruby", "-e", "puts :ok"])
      assert_equal 0, code

      kind, _parser, _hosts, args, _command = MLX::DistributedUtils.instance_variable_get(:@captures).last
      assert_equal :jaccl, kind
      assert_equal "jaccl", args[:backend]
      assert_includes args[:env], "HOSTFILE_ENV=1"
    end
  end

  def test_main_requires_script_or_command
    err = assert_raises(ArgumentError) do
      MLX::DistributedUtils.main(["--backend", "ring"])
    end
    assert_match(/No script is provided/, err.message)
  end
end
