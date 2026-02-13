# frozen_string_literal: true

require "json"
require_relative "test_helper"

$LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
require "mlx"

class Phase203DistributedUtilsCommonParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_hostfile_from_list_and_to_json
    hostfile = MLX::DistributedUtils::Hostfile.from_list("127.0.0.1,worker", 2)
    assert_equal 4, hostfile.hosts.length
    assert_equal "127.0.0.1", hostfile.hosts.first.ssh_hostname
    assert_equal ["127.0.0.1"], hostfile.hosts.first.ips

    hostfile.backend = "ring"
    hostfile.envs = ["A=1"]
    payload = hostfile.to_json_obj
    assert_equal "ring", payload["backend"]
    assert_equal ["A=1"], payload["envs"]
    assert_equal 4, payload["hosts"].length
  end

  def test_hostfile_from_file_parses_formats
    Dir.mktmpdir("mlx-hostfile") do |dir|
      path = File.join(dir, "hosts.json")
      File.write(
        path,
        JSON.dump(
          {
            "backend" => "jaccl",
            "envs" => ["X=1"],
            "hosts" => [{ "ssh" => "h1", "ips" => ["10.0.0.1"], "rdma" => [nil] }]
          }
        )
      )
      parsed = MLX::DistributedUtils::Hostfile.from_file(path)
      assert_equal "jaccl", parsed.backend
      assert_equal ["X=1"], parsed.envs
      assert_equal "h1", parsed.hosts.first.ssh_hostname
    end
  end

  def test_positive_number_and_optional_bool_action
    assert_equal 3, MLX::DistributedUtils.positive_number("3")
    assert_raises(ArgumentError) { MLX::DistributedUtils.positive_number("0") }

    action = MLX::DistributedUtils::OptionalBoolAction.new
    ns = {}
    action.call(nil, ns, nil, "--feature")
    assert_equal true, ns[:feature]
    action.call(nil, ns, nil, "--no-feature")
    assert_equal false, ns[:feature]
  end
end
