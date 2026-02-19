# frozen_string_literal: true

require_relative "../test_helper"

class WebDemoWeightsGuardrailTest < Minitest::Test
  LANGUAGE_DEMO_MAIN_JS = {
    "gpt2" => File.join(RUBY_ROOT, "web", "demo", "gpt2", "main.js"),
    "nanogpt" => File.join(RUBY_ROOT, "web", "demo", "nanogpt", "main.js")
  }.freeze

  CVAE_INDEX_HTML = File.join(RUBY_ROOT, "web", "demo", "cvae", "index.html")
  CVAE_MAIN_JS = File.join(RUBY_ROOT, "web", "demo", "cvae", "main.js")
  GPT2_EXPORTER = File.join(RUBY_ROOT, "tasks", "web_assets_task", "export_gpt2_assets.rb")
  NANOGPT_EXPORTER = File.join(RUBY_ROOT, "tasks", "web_assets_task", "export_nanogpt_assets.rb")

  def test_language_demos_disable_generate_until_weights_are_ready
    LANGUAGE_DEMO_MAIN_JS.each_value do |path|
      source = File.read(path)
      assert(
        source.include?("setGenerationEnabled(false)") || source.include?("generateButton.disabled = true"),
        "#{path} should disable generate while weights/session are unavailable"
      )
    end
  end

  def test_language_demos_surface_missing_weights_message
    LANGUAGE_DEMO_MAIN_JS.each_value do |path|
      source = File.read(path)
      assert_includes source, "Weights: missing"
      refute_includes source, "Weights: random init"
    end
  end

  def test_cvae_demo_has_weights_badge_and_missing_weights_path
    index = File.read(CVAE_INDEX_HTML)
    main = File.read(CVAE_MAIN_JS)

    assert_includes index, 'id="badge-weights"'
    assert_includes main, "const weightsBadge = document.getElementById(\"badge-weights\")"
    assert_includes main, "Weights: missing"
  end

  def test_nanogpt_exporter_no_longer_emits_random_init_mode
    source = File.read(NANOGPT_EXPORTER)
    refute_includes source, "random init"
    assert_includes source, "missing trained artifacts"
  end

  def test_gpt2_exporter_defaults_to_openai_community_repo
    source = File.read(GPT2_EXPORTER)
    assert_includes source, 'DEFAULT_HF_REPO_ID = "openai-community/gpt2"'
    refute_includes source, "hf-internal-testing/tiny-random-gpt2"
  end
end
