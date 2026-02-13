# frozen_string_literal: true

require_relative "test_helper"

class Phase89ArrayStateProtocolTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_copy_and_state_protocol_surface
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)

    %i[__copy__ __deepcopy__ __getstate__ __setstate__ __format__].each do |name|
      assert_respond_to x, name
    end

    copy = x.__copy__
    deep = x.__deepcopy__({})
    assert_equal x.to_a, copy.to_a
    assert_equal x.to_a, deep.to_a

    state = x.__getstate__
    assert_kind_of Hash, state
    restored = x.__setstate__(state)
    assert_equal x.to_a, restored.to_a

    formatted = x.__format__("%0.1f")
    assert_kind_of String, formatted
    assert_includes formatted, "1.0"
  end
end
