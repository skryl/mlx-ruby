# frozen_string_literal: true

require_relative "test_helper"

class Phase0ManifestTest < Minitest::Test
  def test_generates_manifest_with_expected_surface
    script = File.join(RUBY_ROOT, "tools", "parity", "generate_parity_manifest.rb")

    Dir.mktmpdir("mlx-ruby-manifest") do |dir|
      manifest_path = File.join(dir, "parity_manifest.json")
      stdout, stderr, status = Open3.capture3(
        "ruby",
        script,
        "--repo-root",
        REPO_ROOT,
        "--output",
        manifest_path
      )

      assert status.success?, "manifest generation failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert File.exist?(manifest_path), "manifest not written at #{manifest_path}"

      manifest = JSON.parse(File.read(manifest_path))

      assert manifest.dig("metadata", "generated_at")
      assert_equal REPO_ROOT, manifest.dig("metadata", "repo_root")

      assert_operator manifest.dig("python_binding", "defs", "total").to_i, :>=, 400
      assert_operator manifest.dig("python_binding", "defs", "m_def_total").to_i, :>=, 250

      functions = manifest.dig("python_binding", "symbols", "functions") || []
      %w[reshape value_and_grad all_sum export_function].each do |name|
        assert_includes functions, name
      end

      assert_operator manifest.dig("python_package", "nn", "class_count").to_i, :>=, 70
      assert_operator manifest.dig("python_tests", "total_test_cases").to_i, :>=, 650
      assert_operator manifest.dig("python_tests", "by_file", "test_ops.py").to_i, :>=, 100
    end
  end

  def test_contract_checker_accepts_generated_manifest
    generator = File.join(RUBY_ROOT, "tools", "parity", "generate_parity_manifest.rb")
    checker = File.join(RUBY_ROOT, "tools", "parity", "check_parity_manifest.rb")

    Dir.mktmpdir("mlx-ruby-contract") do |dir|
      manifest_path = File.join(dir, "parity_manifest.json")
      _, _, gen_status = Open3.capture3(
        "ruby",
        generator,
        "--repo-root",
        REPO_ROOT,
        "--output",
        manifest_path
      )
      assert gen_status.success?, "generator failed before checker could run"

      stdout, stderr, check_status = Open3.capture3(
        "ruby",
        checker,
        "--manifest",
        manifest_path
      )

      assert check_status.success?, "contract checker failed\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert_match(/Parity contract OK/, stdout)
    end
  end
end
