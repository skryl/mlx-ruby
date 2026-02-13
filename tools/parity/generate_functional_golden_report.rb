#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "pathname"
require "time"

REPO_ROOT = Pathname.new(File.expand_path("../..", __dir__)).freeze
RUBY_ROOT = REPO_ROOT.join("lib").freeze
$LOAD_PATH.unshift(RUBY_ROOT.to_s)
require "mlx"

def nested_close?(expected, actual, atol = 1e-4)
  expected.flatten.zip(actual.flatten).all? { |e, a| (e - a).abs <= atol }
end

def run_check(name)
  pass = yield
  { "name" => name, "pass" => !!pass }
rescue StandardError => e
  { "name" => name, "pass" => false, "error" => "#{e.class}: #{e.message}" }
end

sections = {}

optimizer_checks = []
optimizer_checks << run_check("sgd_single_step") do
  opt = MLX::Optimizers::SGD.new(learning_rate: 0.1)
  p = MLX::Core.array([1.0], MLX::Core.float32)
  g = MLX::Core.array([0.5], MLX::Core.float32)
  new_p = opt.apply_gradients(g, p)
  nested_close?([0.95], new_p.to_a, 1e-5)
end
optimizer_checks << run_check("cross_entropy_known_case") do
  logits = MLX::Core.array([[2.0, 0.0]], MLX::Core.float32)
  target = MLX::Core.array([0], MLX::Core.int32)
  loss = MLX::NN.cross_entropy(logits, target)
  nested_close?([0.12692805], [loss.to_a], 1e-4)
end
optimizer_checks << run_check("module_update_roundtrip") do
  mod = MLX::NN::Module.new
  mod.weight = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
  mod.update({ "weight" => MLX::Core.array([3.0, 4.0], MLX::Core.float32) }, strict: true)
  nested_close?([3.0, 4.0], mod.weight.to_a, 1e-6)
end
sections["optimizer_loss_module"] = {
  "checks" => optimizer_checks,
  "all_pass" => optimizer_checks.all? { |c| c["pass"] }
}

layer_checks = []
layer_checks << run_check("linear_forward_shape") do
  layer = MLX::NN::Linear.new(3, 2)
  out = layer.call(MLX::Core.zeros([4, 3], MLX::Core.float32))
  out.shape == [4, 2]
end
layer_checks << run_check("conv2d_known_value") do
  conv = MLX::NN::Conv2d.new(1, 1, [2, 2], bias: false)
  conv.weight = MLX::Core.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], MLX::Core.float32)
  x = MLX::Core.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], MLX::Core.float32)
  nested_close?([[[[10.0]]]], conv.call(x).to_a, 1e-5)
end
layer_checks << run_check("transformer_shape") do
  t = MLX::NN::Transformer.new(dims: 4, num_heads: 2, num_encoder_layers: 1, num_decoder_layers: 1, mlp_dims: 8)
  src = MLX::Core.zeros([1, 3, 4], MLX::Core.float32)
  tgt = MLX::Core.zeros([1, 2, 4], MLX::Core.float32)
  m1 = MLX::NN::MultiHeadAttention.create_additive_causal_mask(3)
  m2 = MLX::NN::MultiHeadAttention.create_additive_causal_mask(2)
  out = t.call(src, tgt, m1, m2, nil)
  out.shape == [1, 2, 4]
end
layer_checks << run_check("upsample_nearest_shape") do
  up = MLX::NN::Upsample.new(scale_factor: 2, mode: "nearest")
  out = up.call(MLX::Core.zeros([1, 3, 2], MLX::Core.float32))
  out.shape == [1, 6, 2]
end
sections["layers"] = {
  "checks" => layer_checks,
  "all_pass" => layer_checks.all? { |c| c["pass"] }
}

distributed_checks = []
distributed_checks << run_check("hostfile_from_list") do
  hf = MLX::DistributedUtils::Hostfile.from_list("127.0.0.1,worker", 1)
  hf.hosts.length == 2
end
distributed_checks << run_check("dlpack_roundtrip") do
  x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
  capsule = x.__dlpack__
  y = MLX::Core.from_dlpack(capsule)
  capsule.is_a?(MLX::Core::DLPackCapsule) && nested_close?(x.to_a, y.to_a, 1e-6)
end
distributed_checks << run_check("config_parse_ports") do
  parsed = MLX::DistributedUtils.parse_hardware_ports("Hardware Port: Thunderbolt 1\nDevice: en5\n")
  parsed["Thunderbolt 1"] == "en5"
end
distributed_checks << run_check("config_main_dispatch_ethernet") do
  singleton = MLX::DistributedUtils.singleton_class
  singleton.class_eval do
    alias_method :_golden_check_ssh_connections, :check_ssh_connections
    alias_method :_golden_prepare_ethernet_hostfile, :prepare_ethernet_hostfile
    alias_method :_golden_prepare_tb_hostfile, :prepare_tb_hostfile
  end
  begin
    singleton.class_eval do
      define_method(:check_ssh_connections) do |hosts, runner: nil|
        _ = runner
        hosts.map { MLX::DistributedUtils::SSHInfo.new(can_ssh: true, has_sudo: true) }
      end
      define_method(:prepare_ethernet_hostfile) do |_args, _hosts, runner: nil|
        _ = runner
        true
      end
      define_method(:prepare_tb_hostfile) do |_args, _hosts, _sshinfo, runner: nil|
        _ = runner
        true
      end
    end
    MLX::DistributedUtils.__send__(:config_main, ["--over", "ethernet", "--hosts", "127.0.0.1"]) == 0
  ensure
    singleton.class_eval do
      remove_method :check_ssh_connections
      remove_method :prepare_ethernet_hostfile
      remove_method :prepare_tb_hostfile
      alias_method :check_ssh_connections, :_golden_check_ssh_connections
      alias_method :prepare_ethernet_hostfile, :_golden_prepare_ethernet_hostfile
      alias_method :prepare_tb_hostfile, :_golden_prepare_tb_hostfile
      remove_method :_golden_check_ssh_connections
      remove_method :_golden_prepare_ethernet_hostfile
      remove_method :_golden_prepare_tb_hostfile
    end
  end
end
distributed_checks << run_check("launch_script_generation") do
  script = MLX::DistributedUtils::RemoteProcess.make_launch_script(0, nil, {}, [], %w[python train.py], true)
  script.include?("MLX_RANK=0")
end
distributed_checks << run_check("launch_main_print_python") do
  MLX::DistributedUtils.main(["--print-python"]) == 0
end
distributed_checks << run_check("shard_inplace_single_rank") do
  mod = MLX::NN::Module.new
  mod.weight = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
  MLX::NN.shard_inplace(mod, "all-to-sharded")
  nested_close?([1.0, 2.0], mod.weight.to_a, 1e-6)
end
sections["distributed_utils"] = {
  "checks" => distributed_checks,
  "all_pass" => distributed_checks.all? { |c| c["pass"] }
}

uncovered = []
sections.each do |section_name, section|
  section["checks"].each do |check|
    next if check["pass"]

    uncovered << {
      "section" => section_name,
      "check" => check["name"],
      "error" => check["error"]
    }
  end
end

report = {
  "generated_at" => Time.now.utc.iso8601,
  "sections" => sections,
  "all_pass" => sections.values.all? { |s| s["all_pass"] },
  "uncovered_behavior" => uncovered
}

out_file = REPO_ROOT.join("tools", "parity", "reports", "functional_golden_report.json")
File.write(out_file, JSON.pretty_generate(report) + "\n")
puts "wrote #{out_file}"
