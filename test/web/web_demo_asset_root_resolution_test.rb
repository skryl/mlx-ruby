# frozen_string_literal: true

require_relative "../support/test_helper"

class WebDemoAssetRootResolutionTest < Minitest::Test
  DEMO_MAIN_FILES = {
    "gpt2" => File.join(RUBY_ROOT, "web", "demo", "gpt2", "main.js"),
    "nanogpt" => File.join(RUBY_ROOT, "web", "demo", "nanogpt", "main.js"),
    "stable_diffusion" => File.join(RUBY_ROOT, "web", "demo", "stable_diffusion", "main.js")
  }.freeze

  def test_demo_asset_roots_support_multiple_hosting_layouts
    DEMO_MAIN_FILES.each do |name, path|
      source = File.read(path)

      assert_includes source, "const ASSET_ROOT_CANDIDATES = Array.from("
      assert_includes source, "new URL(\"../../assets/#{name}\", import.meta.url)"
      assert_includes source, "new URL(\"../assets/#{name}\", import.meta.url)"
      assert_includes source, "new URL(\"./assets/#{name}\", import.meta.url)"
      assert_includes source, "async function resolveAssetRoot()"
      assert_includes source, "assetExists(`${candidate}/meta.json`)"
      assert_includes source, "missingAssetsMessage"
    end
  end
end
