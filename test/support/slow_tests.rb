# frozen_string_literal: true

module TestSupport
  module SlowTests
    SLOW_TEST_REGISTRY_PATH = File.join(RUBY_ROOT, "test", "slow_tests.json").freeze
    DEFAULT_SLOW_TEST_THRESHOLD_SECONDS = 30.0
    FORCED_SLOW_TEST_PREFIXES = [
      "Phase310OnnxWebgpuHarnessSmokeParityTest#",
      "Phase311OnnxWebgpuHarnessRealRuntimeSmokeParityTest#",
      "Phase313ModelFixtureWebgpuBrowserParityTest#"
    ].freeze
    FORCED_SLOW_TEST_PATH_PREFIXES = [
      File.join(RUBY_ROOT, "test", "integration", "web"),
      File.join(RUBY_ROOT, "test", "integration", "onnx")
    ].freeze

    def include_slow_tests?
      ENV["MLX_TEST_INCLUDE_SLOW"] == "1"
    end

    def forced_slow_test?(test_id)
      FORCED_SLOW_TEST_PREFIXES.any? { |prefix| test_id.start_with?(prefix) }
    end

    def forced_slow_test_path?(source_path)
      return false if source_path.nil?

      expanded = File.expand_path(source_path.to_s)
      FORCED_SLOW_TEST_PATH_PREFIXES.any? do |prefix|
        expanded.start_with?(prefix + File::SEPARATOR) || expanded == prefix
      end
    end

    def slow_test_entry(test_id, source_path: nil)
      return {"forced" => true} if forced_slow_test?(test_id)
      return {"forced" => true} if forced_slow_test_path?(source_path)

      slow_test_registry[test_id]
    end

    def slow_test_registry
      payload = slow_test_payload
      tests = payload["tests"]
      return {} unless tests.is_a?(Hash)

      tests
    end

    def slow_test_threshold_seconds
      payload = slow_test_payload
      raw = payload["threshold_seconds"]
      return DEFAULT_SLOW_TEST_THRESHOLD_SECONDS if raw.nil?

      raw.to_f
    rescue StandardError
      DEFAULT_SLOW_TEST_THRESHOLD_SECONDS
    end

    def slow_test_payload
      return @slow_test_payload if defined?(@slow_test_payload)

      @slow_test_payload = load_slow_test_payload
    end

    def reset_slow_test_payload_cache!
      remove_instance_variable(:@slow_test_payload) if instance_variable_defined?(:@slow_test_payload)
    end

    def load_slow_test_payload
      return {} unless File.file?(SLOW_TEST_REGISTRY_PATH)

      payload = JSON.parse(File.binread(SLOW_TEST_REGISTRY_PATH))
      return payload if payload.is_a?(Hash)

      {}
    rescue JSON::ParserError => e
      warn "failed to parse slow test registry at #{SLOW_TEST_REGISTRY_PATH}: #{e.message}"
      {}
    rescue StandardError => e
      warn "failed to load slow test registry at #{SLOW_TEST_REGISTRY_PATH}: #{e.message}"
      {}
    end
  end
end
