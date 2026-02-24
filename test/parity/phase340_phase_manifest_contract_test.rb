# frozen_string_literal: true

require "yaml"
require_relative "test_helper"
require_relative "../support/parity/phase_manifest_builder"

class Phase340PhaseManifestContractTest < Minitest::Test
  MANIFEST_PATH = File.join(RUBY_ROOT, "test", "parity", "manifest.yml").freeze

  def test_phase_manifest_matches_phase_test_inventory
    assert File.exist?(MANIFEST_PATH), "expected parity phase manifest at #{MANIFEST_PATH}"

    manifest = YAML.safe_load(File.binread(MANIFEST_PATH), permitted_classes: [Time], aliases: false)
    assert_equal "mlx_parity_phase_manifest_v1", manifest.fetch("format")
    assert manifest["generated_at"], "manifest missing generated_at"

    expected = PhaseManifestBuilder.build(repo_root: RUBY_ROOT)
    assert_equal expected.fetch("phases"), manifest.fetch("phases")
  end

  def test_phase_manifest_domains_are_known
    manifest = YAML.safe_load(File.binread(MANIFEST_PATH), permitted_classes: [Time], aliases: false)
    known_domains = %w[core nn optimizers distributed perf]

    manifest.fetch("phases").each do |phase_id, entry|
      assert_includes known_domains, entry.fetch("domain"), "unknown domain for phase #{phase_id}"
      refute_empty entry.fetch("methods"), "phase #{phase_id} must list at least one test method"
    end
  end
end
