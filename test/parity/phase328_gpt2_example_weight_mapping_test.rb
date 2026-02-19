# frozen_string_literal: true

require "json"
require "tmpdir"

require_relative "test_helper"

class Phase328Gpt2ExampleWeightMappingTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    require_relative "../../examples/web/gpt2_example"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_load_hf_state_dict_maps_conv1d_weights_to_mlx_linear
    config = Gpt2Example::Config.new(
      vocab_size: 16,
      n_positions: 8,
      n_embd: 8,
      n_layer: 1,
      n_head: 2,
      dropout: 0.0
    )
    model = Gpt2Example::Gpt2Model.new(config)

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

  def test_forward_shape_matches_gpt2_logits
    config = Gpt2Example::Config.new(
      vocab_size: 32,
      n_positions: 12,
      n_embd: 12,
      n_layer: 2,
      n_head: 3,
      dropout: 0.0
    )
    model = Gpt2Example::Gpt2Model.new(config)
    model.load_hf_state_dict!(build_fake_hf_state(config))

    input_ids = MLX::Core.array([[1, 2, 3, 4, 5, 6]], MLX::Core.int32)
    logits = model.call(input_ids)
    MLX::Core.eval(logits)

    assert_equal [1, 6, 32], logits.shape
  end

  def test_from_hf_directory_falls_back_to_npz_when_native_safetensors_is_disabled
    config_payload = {
      "vocab_size" => 16,
      "n_positions" => 8,
      "n_embd" => 8,
      "n_layer" => 1,
      "n_head" => 2,
      "embd_pdrop" => 0.0,
      "layer_norm_epsilon" => 1e-5
    }
    config = Gpt2Example::Config.from_hf_config(config_payload)
    state = build_fake_hf_state(config)

    Dir.mktmpdir("gpt2_state_test") do |root|
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

      with_singleton_method_stub(Gpt2Example::Gpt2Model, :convert_safetensors_to_npz!, converter_stub) do
        with_singleton_method_stub(MLX::Core, :load, load_stub) do
          model = Gpt2Example::Gpt2Model.from_hf_directory(root)
          assert_instance_of Gpt2Example::Gpt2Model, model
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
    singleton.define_method(method_name, &replacement)
    yield
  ensure
    if had_original
      singleton.define_method(method_name, original_method)
    else
      singleton.send(:remove_method, method_name)
    end
  end

  def build_fake_hf_state(config)
    n = config.n_embd
    hidden = n * 4

    weights = {
      "transformer.wte.weight" => linear_array([config.vocab_size, n], 0.01),
      "transformer.wpe.weight" => linear_array([config.n_positions, n], 0.02),
      "transformer.ln_f.weight" => MLX::Core.ones([n], MLX::Core.float32),
      "transformer.ln_f.bias" => MLX::Core.zeros([n], MLX::Core.float32)
    }

    config.n_layer.times do |layer|
      prefix = "transformer.h.#{layer}"
      weights["#{prefix}.ln_1.weight"] = MLX::Core.ones([n], MLX::Core.float32)
      weights["#{prefix}.ln_1.bias"] = MLX::Core.zeros([n], MLX::Core.float32)
      weights["#{prefix}.ln_2.weight"] = MLX::Core.ones([n], MLX::Core.float32)
      weights["#{prefix}.ln_2.bias"] = MLX::Core.zeros([n], MLX::Core.float32)

      weights["#{prefix}.attn.c_attn.weight"] = linear_array([n, n * 3], 0.001 + layer * 0.0001)
      weights["#{prefix}.attn.c_attn.bias"] = linear_array([n * 3], 0.0005 + layer * 0.0001)
      weights["#{prefix}.attn.c_proj.weight"] = linear_array([n, n], 0.0012 + layer * 0.0001)
      weights["#{prefix}.attn.c_proj.bias"] = linear_array([n], 0.0007 + layer * 0.0001)

      weights["#{prefix}.mlp.c_fc.weight"] = linear_array([n, hidden], 0.0008 + layer * 0.0001)
      weights["#{prefix}.mlp.c_fc.bias"] = linear_array([hidden], 0.0004 + layer * 0.0001)
      weights["#{prefix}.mlp.c_proj.weight"] = linear_array([hidden, n], 0.0009 + layer * 0.0001)
      weights["#{prefix}.mlp.c_proj.bias"] = linear_array([n], 0.0006 + layer * 0.0001)
    end

    # HF GPT-2 ties lm_head to token embeddings.
    weights["lm_head.weight"] = weights.fetch("transformer.wte.weight")
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
