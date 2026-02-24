# frozen_string_literal: true

require "json"
require "timeout"
require "minitest/autorun"
require "open3"
require "rbconfig"
require "tmpdir"
require "fileutils"

default_ruby_root = File.expand_path("../..", __dir__)
RUBY_ROOT = File.expand_path(ENV.fetch("MLX_TEST_RUBY_ROOT", default_ruby_root))
default_repo_root = File.expand_path("..", RUBY_ROOT)
REPO_ROOT = File.expand_path(ENV.fetch("MLX_TEST_REPO_ROOT", default_repo_root))

require_relative "native_build"
require_relative "slow_tests"
require_relative "export_helpers"
require_relative "tmp_helpers"
require_relative "parity_paths"

module TestSupport
  extend NativeBuild
  extend SlowTests
  extend ExportHelpers
  extend TmpHelpers
  extend ParityPaths

  SLOW_TEST_REGISTRY_PATH = SlowTests::SLOW_TEST_REGISTRY_PATH
  DEFAULT_SLOW_TEST_THRESHOLD_SECONDS = SlowTests::DEFAULT_SLOW_TEST_THRESHOLD_SECONDS
  FORCED_SLOW_TEST_PREFIXES = SlowTests::FORCED_SLOW_TEST_PREFIXES
  FORCED_SLOW_TEST_PATH_PREFIXES = SlowTests::FORCED_SLOW_TEST_PATH_PREFIXES
end

begin
  TestSupport.build_native_extension!
rescue StandardError
  nil
end

raw_test_timeout = ENV.fetch("MLX_TEST_TIMEOUT", "10").to_i
TEST_TIMEOUT_SECONDS = raw_test_timeout.positive? ? raw_test_timeout : 10

module Minitest
  class Test
    alias_method :before_setup_without_slow_test_gate, :before_setup
    alias_method :run_without_timeout, :run

    def before_setup
      maybe_skip_slow_test!
      before_setup_without_slow_test_gate
    end

    def run
      Timeout.timeout(self.class.current_test_timeout_seconds) { run_without_timeout }
    end

    def self.current_test_timeout_seconds
      raw = ENV.fetch("MLX_TEST_TIMEOUT", TEST_TIMEOUT_SECONDS.to_s).to_i
      raw.positive? ? raw : TEST_TIMEOUT_SECONDS
    rescue StandardError
      TEST_TIMEOUT_SECONDS
    end

    private

    def maybe_skip_slow_test!
      return if TestSupport.include_slow_tests?

      slow_entry = TestSupport.slow_test_entry(slow_test_identifier, source_path: slow_test_source_path)
      return if slow_entry.nil?

      threshold = TestSupport.slow_test_threshold_seconds
      max_seconds = slow_entry["max_seconds"]
      measured = max_seconds.nil? ? "" : format(" (measured %.2fs)", max_seconds.to_f)
      skip "slow test (>#{threshold}s#{measured}); run `rake test:all` or set MLX_TEST_INCLUDE_SLOW=1"
    end

    def slow_test_identifier
      "#{self.class}##{name}"
    end

    def slow_test_source_path
      self.class.instance_method(name).source_location&.first
    rescue NameError
      nil
    end
  end
end
