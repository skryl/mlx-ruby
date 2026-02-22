# frozen_string_literal: true

require_relative "test_helper"

class SlowTestGateTest < Minitest::Test
  class SlowGateProbeTest < Minitest::Test
    def test_marked_slow
      assert true
    end
  end

  def setup
    @singleton = class << TestSupport
      self
    end
    @restores = []
  end

  def teardown
    @restores.reverse_each do |name, backup|
      @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
      @singleton.alias_method(name, backup)
      @singleton.remove_method(backup)
    end
  end

  def test_before_setup_skips_marked_slow_tests_when_fast_mode_is_enabled
    test_id = "SlowTestGateTest::SlowGateProbeTest#test_marked_slow"
    stub_singleton_method(:slow_test_entry) do |candidate|
      candidate == test_id ? {"max_seconds" => 45.7} : nil
    end
    stub_singleton_method(:include_slow_tests?) { false }

    probe = SlowGateProbeTest.new(:test_marked_slow)
    error = assert_raises(Minitest::Skip) { probe.before_setup }
    assert_match(/slow test/i, error.message)
    assert_match(/rake test:all/i, error.message)
  end

  def test_before_setup_allows_marked_slow_tests_when_include_slow_is_enabled
    test_id = "SlowTestGateTest::SlowGateProbeTest#test_marked_slow"
    stub_singleton_method(:slow_test_entry) do |candidate|
      candidate == test_id ? {"max_seconds" => 45.7} : nil
    end
    stub_singleton_method(:include_slow_tests?) { true }

    probe = SlowGateProbeTest.new(:test_marked_slow)
    assert_nil probe.before_setup
  end

  private

  def stub_singleton_method(name, &block)
    backup = :"__slow_gate_restore_#{name}_#{@restores.length}"
    @singleton.alias_method(backup, name)
    @restores << [name, backup]
    @singleton.remove_method(name) if @singleton.instance_methods(false).include?(name)
    @singleton.define_method(name, &block)
  end
end
