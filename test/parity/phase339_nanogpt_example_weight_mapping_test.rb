# frozen_string_literal: true

require "json"
require "tmpdir"

require_relative "test_helper"

class Phase339NanoGptExampleWeightMappingTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    require_relative "../../examples/web/nanogpt_example"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_config_from_hf_config_parses_nanogpt_shakespeare_payload
    payload = {
      "vocab_size" => 65,
      "block_size" => 256,
      "n_layer" => 6,
      "n_head" => 6,
      "n_embd" => 384,
      "dropout" => 0.2,
      "bias" => false
    }
    config = NanoGptExample::Config.from_hf_config(payload)

    assert_equal 65, config.vocab_size
    assert_equal 256, config.block_size
    assert_equal 6, config.n_layer
    assert_equal 6, config.n_head
    assert_equal 384, config.n_embd
    assert_in_delta 0.2, config.dropout, 1e-9
    assert_equal false, config.bias
  end

  def test_load_hf_state_dict_maps_qkv_and_mlp_weights
    config = NanoGptExample::Config.new(
      vocab_size: 16,
      block_size: 8,
      n_layer: 1,
      n_head: 2,
      n_embd: 8,
      dropout: 0.0,
      bias: false
    )
    model = NanoGptExample::NanoGptModel.new(
      vocab_size: config.vocab_size,
      block_size: config.block_size,
      n_layer: config.n_layer,
      n_head: config.n_head,
      n_embd: config.n_embd,
      dropout: config.dropout,
      bias: config.bias
    )

    state = build_fake_hf_state(config)
    model.load_hf_state_dict!(state)

    fc_expected = transpose_2d(state.fetch("transformer.h.0.mlp.c_fc.weight").to_a)
    fc_actual = model.blocks[0].mlp.fc.weight.to_a
    assert_nested_close(fc_expected, fc_actual)

    c_attn = state.fetch("transformer.h.0.attn.c_attn.weight")
    q_expected = transpose_2d(MLX::Core.slice(c_attn, [0, 0], [config.n_embd, config.n_embd]).to_a)
    k_expected = transpose_2d(MLX::Core.slice(c_attn, [0, config.n_embd], [config.n_embd, config.n_embd * 2]).to_a)
    v_expected = transpose_2d(
      MLX::Core.slice(c_attn, [0, config.n_embd * 2], [config.n_embd, config.n_embd * 3]).to_a
    )

    assert_nested_close(q_expected, model.blocks[0].attn.q_proj.weight.to_a)
    assert_nested_close(k_expected, model.blocks[0].attn.k_proj.weight.to_a)
    assert_nested_close(v_expected, model.blocks[0].attn.v_proj.weight.to_a)
  end

  def test_from_hf_directory_falls_back_to_npz_when_native_safetensors_is_disabled
    config_payload = {
      "vocab_size" => 16,
      "block_size" => 8,
      "n_layer" => 1,
      "n_head" => 2,
      "n_embd" => 8,
      "dropout" => 0.0,
      "bias" => false
    }
    config = NanoGptExample::Config.from_hf_config(config_payload)
    state = build_fake_hf_state(config)

    Dir.mktmpdir("nanogpt_state_test") do |root|
      File.binwrite(File.join(root, "config.json"), JSON.generate(config_payload))
      File.binwrite(File.join(root, "model.safetensors"), "stub")
      fallback_npz = File.join(root, "model.npz")
      safetensors_path = File.join(root, "model.safetensors")

      fake_loaded = FakeLoadedState.new(state)
      load_paths = []
      load_stub = lambda do |path|
        load_paths << path
        if path.end_with?(".safetensors")
          raise "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON to enable safetensors support."
        end

        fake_loaded
      end

      converter_stub = lambda do |_source, destination|
        File.binwrite(destination, "npz")
        destination
      end

      with_singleton_method_stub(NanoGptExample::NanoGptModel, :convert_safetensors_to_npz!, converter_stub) do
        with_singleton_method_stub(MLX::Core, :load, load_stub) do
          model = NanoGptExample::NanoGptModel.from_hf_directory(root)
          assert_instance_of NanoGptExample::NanoGptModel, model
          assert_equal [safetensors_path, fallback_npz], load_paths
        end
      end
    end
  end

  private

  FakeLoadedState = Struct.new(:state) do
    def to_a
      state.to_a
    end
  end

  def with_singleton_method_stub(target, method_name, replacement)
    singleton = target.singleton_class
    had_original = singleton.method_defined?(method_name)
    original_method = singleton.instance_method(method_name) if had_original
    stubbed = false

    singleton.send(:remove_method, method_name) if had_original
    singleton.send(:define_method, method_name, &replacement)
    stubbed = true
    yield
  ensure
    singleton.send(:remove_method, method_name) if stubbed

    if had_original && original_method
      singleton.define_method(method_name, original_method)
    end
  end

  def build_fake_hf_state(config)
    n = config.n_embd
    hidden = n * 4

    weights = {
      "transformer.wte.weight" => linear_array([config.vocab_size, n], 0.01),
      "transformer.wpe.weight" => linear_array([config.block_size, n], 0.02),
      "transformer.ln_f.weight" => MLX::Core.ones([n], MLX::Core.float32),
      "lm_head.weight" => linear_array([config.vocab_size, n], 0.03)
    }

    config.n_layer.times do |layer|
      prefix = "transformer.h.#{layer}"
      weights["#{prefix}.ln_1.weight"] = MLX::Core.ones([n], MLX::Core.float32)
      weights["#{prefix}.ln_2.weight"] = MLX::Core.ones([n], MLX::Core.float32)

      weights["#{prefix}.attn.c_attn.weight"] = linear_array([n, n * 3], 0.001 + layer * 0.0001)
      weights["#{prefix}.attn.c_proj.weight"] = linear_array([n, n], 0.0012 + layer * 0.0001)

      weights["#{prefix}.mlp.c_fc.weight"] = linear_array([n, hidden], 0.0008 + layer * 0.0001)
      weights["#{prefix}.mlp.c_proj.weight"] = linear_array([hidden, n], 0.0009 + layer * 0.0001)
    end

    weights
  end

  def linear_array(shape, scale)
    total = shape.reduce(1) { |acc, dim| acc * dim }
    values = Array.new(total) { |idx| idx.to_f * scale.to_f }
    MLX::Core.reshape(MLX::Core.array(values, MLX::Core.float32), shape)
  end

  def transpose_2d(matrix)
    return [] if matrix.empty?

    rows = matrix.length
    cols = matrix[0].length
    Array.new(cols) do |c|
      Array.new(rows) { |r| matrix[r][c] }
    end
  end

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal signature(expected), signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp.to_f, got.to_f, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| signature(v) })]
  end
end
