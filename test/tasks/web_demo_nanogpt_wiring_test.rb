# frozen_string_literal: true

require_relative "../test_helper"

class WebDemoNanogptWiringTest < Minitest::Test
  NANOGPT_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "nanogpt", "main.js")
  NANOGPT_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "nanogpt", "index.html")

  def test_nanogpt_demo_uses_nanogpt_assets_and_tokenizer
    source = File.read(NANOGPT_MAIN_JS)

    assert_includes source, 'const ASSET_ROOT = "../../assets/nanogpt"'
    assert_includes source, "const TOKENIZER_PATH = `${ASSET_ROOT}/tokenizer.json`"
    refute_includes source, "const VOCAB_PATH"
    refute_includes source, "const MERGES_PATH"
  end

  def test_nanogpt_demo_has_distinct_nanogpt_title
    source = File.read(NANOGPT_INDEX_HTML)

    assert_includes source, "<title>nanoGPT Shakespeare: Web Demo</title>"
    assert_includes source, "<h1>nanoGPT Shakespeare</h1>"
    refute_includes source, "<h1>GPT-2 Sparks</h1>"
  end
end
