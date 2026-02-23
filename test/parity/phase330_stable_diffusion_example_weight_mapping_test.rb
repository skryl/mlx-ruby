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

  def test_map_unet_weights_remaps_core_keys_without_slicing
    ff_proj = rand_tensor([640, 320])
    conv_in = rand_tensor([32, 4, 3, 3])
    attn_key = rand_tensor([128, 768])

    state = {
      "conv_in.weight" => conv_in,
      "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight" => ff_proj,
      "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight" => rand_tensor([320, 1280]),
      "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight" => attn_key
    }

    mapped = StableDiffusionExample.map_unet_weights(state)

    expected_conv = MLX::Core.transpose(conv_in, [0, 2, 3, 1]).to_a
    assert_nested_close(expected_conv, mapped.fetch("conv_in.weight").to_a)

    linear1_key = "down_blocks.0.attentions.0.transformer_blocks.0.linear1.weight"
    linear2_key = "down_blocks.0.attentions.0.transformer_blocks.0.linear2.weight"
    linear3_key = "down_blocks.0.attentions.0.transformer_blocks.0.linear3.weight"
    mapped_attn_key = "mid_blocks.1.transformer_blocks.0.attn2.key_proj.weight"

    assert_equal [320, 320], mapped.fetch(linear1_key).shape
    assert_equal [320, 320], mapped.fetch(linear2_key).shape
    assert_includes mapped.keys, linear3_key

    part1, part2 = MLX::Core.split(ff_proj, 2, 0)
    assert_nested_close(part1.to_a, mapped.fetch(linear1_key).to_a)
    assert_nested_close(part2.to_a, mapped.fetch(linear2_key).to_a)

    assert_equal [128, 768], mapped.fetch(mapped_attn_key).shape
    assert_nested_close(attn_key.to_a, mapped.fetch(mapped_attn_key).to_a)
  end

  def test_map_clip_text_encoder_weights_normalizes_names
    state = {
      "text_model.embeddings.token_embedding.weight" => rand_tensor([49_408, 768]),
      "text_model.embeddings.position_embedding.weight" => rand_tensor([77, 768]),
      "text_model.encoder.layers.0.self_attn.q_proj.weight" => rand_tensor([768, 768]),
      "text_model.encoder.layers.0.mlp.fc1.weight" => rand_tensor([3072, 768]),
      "text_model.final_layer_norm.weight" => rand_tensor([768])
    }

    mapped = StableDiffusionExample.map_clip_text_encoder_weights(state)

    assert_includes mapped.keys, "token_embedding.weight"
    assert_includes mapped.keys, "position_embedding.weight"
    assert_includes mapped.keys, "layers.0.attention.query_proj.weight"
    assert_includes mapped.keys, "layers.0.linear1.weight"
    assert_includes mapped.keys, "final_layer_norm.weight"

    assert_equal [768, 768], mapped.fetch("layers.0.attention.query_proj.weight").shape
    assert_equal [3072, 768], mapped.fetch("layers.0.linear1.weight").shape
  end

  def test_map_vae_weights_remaps_and_transposes_expected_tensors
    downsample_conv = rand_tensor([64, 64, 3, 3])
    quant_conv = rand_tensor([8, 8, 1, 1])

    state = {
      "encoder.down_blocks.0.downsamplers.0.conv.weight" => downsample_conv,
      "mid_block.attentions.0.to_out.0.weight" => rand_tensor([512, 512]),
      "quant_conv.weight" => quant_conv
    }

    mapped = StableDiffusionExample.map_vae_weights(state)

    downsample_key = "encoder.down_blocks.0.downsample.weight"
    out_proj_key = "mid_blocks.1.out_proj.weight"
    quant_proj_key = "quant_proj.weight"

    expected_downsample = MLX::Core.transpose(downsample_conv, [0, 2, 3, 1]).to_a
    assert_nested_close(expected_downsample, mapped.fetch(downsample_key).to_a)

    assert_includes mapped.keys, out_proj_key
    assert_equal [512, 512], mapped.fetch(out_proj_key).shape

    assert_equal [8, 8], mapped.fetch(quant_proj_key).shape
    assert_nested_close(MLX::Core.squeeze(quant_conv).to_a, mapped.fetch(quant_proj_key).to_a)
  end

  private

  def rand_tensor(shape)
    MLX::Core.random_uniform(shape, -1.0, 1.0, MLX::Core.float32)
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
