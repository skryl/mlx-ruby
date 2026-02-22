# frozen_string_literal: true

require_relative "test_helper"

class Phase335RealRuntimeSmokeDependencyGateParityTest < Minitest::Test
  PHASE311_TEST = File.join(__dir__, "phase311_onnx_webgpu_harness_real_runtime_smoke_parity_test.rb")

  def test_phase311_real_runtime_smoke_is_not_opt_in_via_env_gate
    source = File.binread(PHASE311_TEST)

    refute_includes source, "MLX_TEST_WEB_SMOKE_REAL"
  end

  def test_phase311_real_runtime_smoke_is_dependency_gated
    source = File.binread(PHASE311_TEST)

    assert_includes source, 'skip "python onnx module is required for phase311 tests"'
    assert_includes source, 'skip "node is required for phase311 tests"'
    assert_includes source, 'skip "playwright module is required for phase311 tests"'
    assert_includes source, 'skip "onnxruntime-web module is required for phase311 tests"'
  end
end
