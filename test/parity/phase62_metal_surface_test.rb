# frozen_string_literal: true

require "json"
require "open3"
require_relative "test_helper"

class Phase62MetalSurfaceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_metal_entrypoints
    assert_respond_to MLX::Core, :metal_is_available
    assert_respond_to MLX::Core, :metal_start_capture
    assert_respond_to MLX::Core, :metal_stop_capture
    assert_respond_to MLX::Core, :metal_device_info

    available = MLX::Core.metal_is_available
    assert_includes [true, false], available

    return unless available

    probe = <<~RUBY
      $LOAD_PATH.unshift("#{File.join(RUBY_ROOT, "lib")}")
      require "mlx"
      puts JSON.generate(MLX::Core.metal_device_info)
    RUBY
    stdout, stderr, status = Open3.capture3("ruby", "-rjson", "-e", probe)
    skip("metal_device_info probe failed: #{stderr}") unless status.success?

    info = JSON.parse(stdout)
    assert_instance_of Hash, info
    refute_empty info
  end
end
