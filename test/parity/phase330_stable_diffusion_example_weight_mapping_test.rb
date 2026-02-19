# frozen_string_literal: true

require_relative "test_helper"

class Phase330StableDiffusionExampleWeightMappingTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    require_relative "../../examples/web/stable_diffusion_example"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_load_hf_state_dict_maps_core_unet_weights
    model = StableDiffusionExample::TinyUnetModel.new(cross_attention_dim: 32)
    state = build_fake_hf_state

    model.load_hf_state_dict!(state)

    expected_conv_in = transpose_conv_weight(state.fetch("conv_in.weight"))
    assert_nested_close(expected_conv_in, model.conv_in.weight.to_a)

    expected_down = transpose_conv_weight(state.fetch("down_blocks.1.resnets.0.conv1.weight"))
    assert_nested_close(expected_down, model.down.weight.to_a)

    expected_cond_proj = MLX::Core.slice(
      state.fetch("down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight"),
      [0, 0],
      [32, 32]
    ).to_a
    assert_nested_close(expected_cond_proj, model.cond_proj.weight.to_a)
  end

  def test_forward_shape_matches_unet_like_signature
    model = StableDiffusionExample::TinyUnetModel.new(cross_attention_dim: 32)
    model.load_hf_state_dict!(build_fake_hf_state)

    latents = MLX::Core.random_uniform([1, 8, 8, 4], -1.0, 1.0, MLX::Core.float32)
    timestep = MLX::Core.array([1.0], MLX::Core.float32)
    encoder_hidden_states = MLX::Core.random_uniform([1, 77, 32], -1.0, 1.0, MLX::Core.float32)

    output = model.call(latents, timestep, encoder_hidden_states)
    MLX::Core.eval(output)

    assert_equal [1, 8, 8, 4], output.shape
  end

  private

  def build_fake_hf_state
    {
      "conv_in.weight" => rand_tensor([32, 4, 3, 3]),
      "conv_in.bias" => rand_tensor([32]),
      "down_blocks.1.resnets.0.conv1.weight" => rand_tensor([64, 32, 3, 3]),
      "down_blocks.1.resnets.0.conv1.bias" => rand_tensor([64]),
      "mid_block.resnets.0.conv1.weight" => rand_tensor([64, 64, 3, 3]),
      "mid_block.resnets.0.conv1.bias" => rand_tensor([64]),
      "up_blocks.1.resnets.2.conv1.weight" => rand_tensor([32, 64, 3, 3]),
      "up_blocks.1.resnets.2.conv1.bias" => rand_tensor([32]),
      "conv_out.weight" => rand_tensor([4, 32, 3, 3]),
      "conv_out.bias" => rand_tensor([4]),
      "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight" => rand_tensor([64, 32]),
      "time_embedding.linear_2.weight" => rand_tensor([128, 128]),
      "time_embedding.linear_2.bias" => rand_tensor([128])
    }
  end

  def rand_tensor(shape)
    MLX::Core.random_uniform(shape, -1.0, 1.0, MLX::Core.float32)
  end

  def transpose_conv_weight(weight)
    MLX::Core.transpose(weight, [0, 2, 3, 1]).to_a
  end

  def assert_nested_close(expected, actual, atol: 1e-5)
    if expected.is_a?(Array) && actual.is_a?(Array)
      assert_equal expected.length, actual.length
      expected.each_with_index do |sub_expected, idx|
        assert_nested_close(sub_expected, actual[idx], atol: atol)
      end
      return
    end

    assert_in_delta expected.to_f, actual.to_f, atol
  end
end
