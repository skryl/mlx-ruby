# frozen_string_literal: true

require "minitest/autorun"
require "open3"
require "rbconfig"

class ExtconfCompatibilityGuardTest < Minitest::Test
  REPO_ROOT = File.expand_path("../../..", __dir__)
  EXTCONF_PATH = File.join(REPO_ROOT, "ext", "mlx", "extconf.rb")
  SHA_A = "1111111111111111111111111111111111111111"
  SHA_B = "2222222222222222222222222222222222222222"

  def run_extconf(extra_env = {})
    env = {
      "MLX_EXTCONF_VALIDATE_ONLY" => "1",
      "MLX_EXTCONF_TEST_MLX_REVISION" => SHA_A,
      "MLX_EXTCONF_TEST_MLX_ONNX_PINNED_MLX_REVISION" => SHA_A
    }.merge(extra_env)

    Open3.capture3(env, RbConfig.ruby, EXTCONF_PATH, chdir: REPO_ROOT)
  end

  def test_validate_only_succeeds_when_revisions_match
    stdout, stderr, status = run_extconf

    assert status.success?, "extconf validate-only failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
    assert_match(/mlx-onnx compatibility check passed/, stdout)
  end

  def test_validate_only_fails_when_revisions_differ
    _stdout, stderr, status = run_extconf(
      "MLX_EXTCONF_TEST_MLX_ONNX_PINNED_MLX_REVISION" => SHA_B
    )

    refute status.success?, "extconf validate-only unexpectedly succeeded"
    assert_match(/mlx\/mlx-onnx revision mismatch detected/, stderr)
    assert_match(/git submodule update --init --recursive/, stderr)
  end
end
