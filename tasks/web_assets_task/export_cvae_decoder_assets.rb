# frozen_string_literal: true

require "fileutils"
require "json"
require "stringio"

REPO_ROOT = File.expand_path("../..", __dir__)
LIB_ROOT = File.join(REPO_ROOT, "lib")
$LOAD_PATH.unshift(LIB_ROOT) unless $LOAD_PATH.include?(LIB_ROOT)

require "mlx"
require_relative "../../mlx-ruby-examples/cvae/vae"

module CvaeDecoderWebAssets
  module_function

  OUTPUT_DIR = File.join(REPO_ROOT, "web", "assets", "cvae")
  MODEL_NAME = "cvae_decoder_web_demo"
  LATENT_DIMS = 4
  IMAGE_SHAPE = [64, 64, 1]
  MAX_FILTERS = 32
  RANDOM_SEED = 7
  PRESETS = {
    "calm" => [0.0, 0.0, 0.0, 0.0],
    "drift" => [1.5, -1.0, 0.75, -0.5],
    "shock" => [-2.0, 1.5, -1.25, 2.0],
    "echo" => [0.5, 1.75, -1.75, -0.25]
  }.freeze

  def run!
    unless MLX.native_available?
      abort("MLX native extension unavailable. Run `bundle exec rake build` first.")
    end

    MLX::Core.random_seed(RANDOM_SEED)
    model = CvaeExample::CVAE.new(LATENT_DIMS, IMAGE_SHAPE, MAX_FILTERS)
    # Freeze parameter initialization so export does not include random-init ops.
    MLX::Core.eval(model.parameters)

    latent_seed = MLX::Core.array([PRESETS.fetch("calm")], MLX::Core.float32)
    payload_json = MLX::Core.export_graph_ir(
      StringIO.new,
      ->(latent) { model.decode(latent) },
      latent_seed
    )
    payload = JSON.parse(payload_json)
    report = MLX::Core.graph_ir_webgpu_compatibility_report(payload)
    unsupported_nodes = report.fetch("unsupported_nodes").to_i
    unless unsupported_nodes.zero?
      abort(
        "CVAE decoder graph is not fully WebGPU-compatible. " \
        "unsupported_ops=#{report.fetch('unsupported_ops').inspect}"
      )
    end

    FileUtils.mkdir_p(OUTPUT_DIR)

    graph_ir_path = File.join(OUTPUT_DIR, "graph_ir.json")
    File.binwrite(graph_ir_path, JSON.pretty_generate(payload))

    onnx_path = File.join(OUTPUT_DIR, "model.onnx")
    MLX::Core.export_onnx(onnx_path, payload, model_name: MODEL_NAME)

    input_spec = payload.fetch("inputs").first
    output_spec = payload.fetch("outputs").first

    metadata = {
      "format" => "cvae_decoder_demo_asset_v1",
      "model_name" => MODEL_NAME,
      "seed" => RANDOM_SEED,
      "input" => input_spec,
      "output" => output_spec,
      "webgpu_compatibility" => {
        "unsupported_nodes" => unsupported_nodes,
        "unsupported_ops" => report.fetch("unsupported_ops")
      }
    }

    File.binwrite(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(metadata))
    File.binwrite(File.join(OUTPUT_DIR, "input.presets.json"), JSON.pretty_generate(PRESETS))

    puts "Wrote CVAE decoder demo assets to #{OUTPUT_DIR}"
    puts "  - #{graph_ir_path}"
    puts "  - #{onnx_path}"
    puts "  - #{File.join(OUTPUT_DIR, 'meta.json')}"
    puts "  - #{File.join(OUTPUT_DIR, 'input.presets.json')}"
  end
end

if $PROGRAM_NAME == __FILE__
  CvaeDecoderWebAssets.run!
end
