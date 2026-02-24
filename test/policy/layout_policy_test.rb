# frozen_string_literal: true

require_relative "../support/test_helper"

class TestLayoutPolicyTest < Minitest::Test
  def test_web_integration_tests_live_under_integration_web
    expected_paths = [
      File.join(RUBY_ROOT, "test", "integration", "web", "gpt2_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "integration", "web", "nanogpt_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "integration", "web", "stable_diffusion_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "integration", "web", "web_demo_integration_test_helper.rb"),
      File.join(RUBY_ROOT, "test", "integration", "web", "demo_page_probe.mjs")
    ]
    legacy_paths = [
      File.join(RUBY_ROOT, "test", "web", "gpt2_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "web", "nanogpt_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "web", "stable_diffusion_demo_integration_test.rb"),
      File.join(RUBY_ROOT, "test", "web", "web_demo_integration_test_helper.rb"),
      File.join(RUBY_ROOT, "test", "web", "demo_page_probe.mjs")
    ]

    expected_paths.each do |path|
      assert File.exist?(path), "expected integration web test file to exist: #{path}"
    end

    legacy_paths.each do |path|
      refute File.exist?(path), "expected legacy web test file to be moved: #{path}"
    end
  end

  def test_onnx_integration_tests_live_under_integration_onnx
    filenames = [
      "benchmark_model_onnx_runtime_test.rb",
      "examples_submodule_full_export_test.rb",
      "examples_submodule_onnx_runtime_test.rb",
      "export_onnx_runtime_test.rb",
      "export_onnx_webgpu_harness_test.rb",
      "model_fixture_onnx_runtime_test.rb",
      "model_fixture_webgpu_browser_test.rb",
      "native_onnx_binary_python_oracle_test.rb",
      "onnx_runtime_op_coverage_test.rb",
      "onnx_webgpu_compat_report_test.rb",
      "onnx_webgpu_coverage_artifact_test.rb",
      "onnx_webgpu_coverage_submodule_mode_test.rb",
      "onnx_webgpu_harness_real_runtime_smoke_test.rb",
      "onnx_webgpu_harness_smoke_test.rb",
      "real_runtime_smoke_dependency_gate_test.rb",
      "webgpu_python_metrics_test.rb"
    ]

    expected_paths = filenames.map { |name| File.join(RUBY_ROOT, "test", "integration", "onnx", name) }
    legacy_paths = filenames.map { |name| File.join(RUBY_ROOT, "test", "onnx", name) }

    expected_paths.each do |path|
      assert File.exist?(path), "expected integration onnx test file to exist: #{path}"
    end

    legacy_paths.each do |path|
      refute File.exist?(path), "expected legacy onnx integration test file to be moved: #{path}"
    end
  end

  def test_onnx_unit_tests_live_under_unit_onnx
    legacy_unit_tests = Dir.glob(File.join(RUBY_ROOT, "test", "onnx", "*_test.rb")).sort
    assert_empty legacy_unit_tests, "expected legacy onnx unit tests to be moved under test/unit/onnx"

    expected_unit_paths = [
      File.join(RUBY_ROOT, "test", "unit", "onnx", "onnx_validation_test.rb"),
      File.join(RUBY_ROOT, "test", "unit", "onnx", "onnx_export_contract_boundary_test.rb"),
      File.join(RUBY_ROOT, "test", "unit", "onnx", "argreduce_lowering_test.rb"),
      File.join(RUBY_ROOT, "test", "unit", "onnx", "test_helper.rb")
    ]

    expected_unit_paths.each do |path|
      assert File.exist?(path), "expected unit onnx test file to exist: #{path}"
    end
  end
end
