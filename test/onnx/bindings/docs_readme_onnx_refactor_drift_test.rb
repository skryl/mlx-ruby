# frozen_string_literal: true

require_relative "test_helper"

class Phase332DocsReadmeGraphIrRefactorDriftParityTest < Minitest::Test
  REPO_ROOT = RUBY_ROOT

  PHASE6_DOC_FILES = [
    "README.md",
    "docs/src/ruby/export.rst",
    "docs/src/onnx_webgpu/index.rst",
    "docs/src/onnx_webgpu/mlx_to_onnx.rst",
    "docs/src/onnx_webgpu/onnx_to_onnx.rst",
    "docs/src/onnx_webgpu/validation_and_compatibility.rst",
    "docs/src/onnx_webgpu/webgpu_harness_and_smoke.rst"
  ].freeze

  REQUIRED_OWNERSHIP_TERMS = [
    "MLX::ONNX",
    "MLX::ONNX::Native",
    "MLX::ONNX::WebGPUHarness"
  ].freeze

  REQUIRED_PUBLIC_FLOW_TERMS = [
    "MLX::ONNX.export_graph_ir",
    "MLX::ONNX.export_graph_ir_json",
    "MLX::ONNX.export_onnx_compatibility_report",
    "MLX::ONNX.graph_ir_to_onnx",
    "MLX::ONNX.graph_ir_to_onnx_json",
    "MLX::ONNX.export_onnx",
    "MLX::ONNX.export_onnx_json",
    "MLX::ONNX::WebGPUHarness.export_onnx_webgpu_harness",
    "MLX::ONNX::WebGPUHarness.smoke_test_onnx_webgpu_harness"
  ].freeze

  REQUIRED_OUTPUT_TERMS = [
    "model.onnx",
    "harness.manifest.json",
    "inputs.example.json",
    "index.html",
    "harness.js",
    "onnx_webgpu_harness_v1",
    "onnx_webgpu_telemetry_v1",
    "webgpu_compat_report_v1"
  ].freeze

  def test_docs_and_readme_capture_refactored_ownership_terms
    text = phase6_text_corpus
    REQUIRED_OWNERSHIP_TERMS.each do |term|
      assert_includes text, term, "missing required ownership term: #{term}"
    end
  end

  def test_docs_and_readme_keep_core_facade_flow_and_outputs_documented
    text = phase6_text_corpus
    REQUIRED_PUBLIC_FLOW_TERMS.each do |term|
      assert_includes text, term, "missing required MLX::ONNX flow term: #{term}"
    end
    REQUIRED_OUTPUT_TERMS.each do |term|
      assert_includes text, term, "missing required output/format term: #{term}"
    end
  end

  private

  def phase6_text_corpus
    @phase6_text_corpus ||= PHASE6_DOC_FILES.map do |relative_path|
      absolute_path = File.join(REPO_ROOT, relative_path)
      File.binread(absolute_path)
    end.join("\n")
  end
end
