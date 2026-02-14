# frozen_string_literal: true

require_relative "mlx/version"

module MLX
  class << self
    def native_available?
      @native_available == true
    end
  end

  @native_available = false

  begin
    require_relative "../ext/mlx/native"
  rescue LoadError
    begin
      require "mlx/native"
    rescue LoadError
      nil
    end
  end

  @native_available = defined?(MLX::Native) &&
      MLX::Native.respond_to?(:loaded?) &&
      MLX::Native.loaded?
end

require_relative "mlx/core"

if MLX.native_available? &&
    defined?(MLX::Core) &&
    MLX::Core.respond_to?(:set_default_device) &&
    MLX::Core.respond_to?(:cpu) &&
    MLX::Core.respond_to?(:gpu)
  begin
    requested_device = ENV.fetch("MLX_DEFAULT_DEVICE", ENV.fetch("DEVICE", "cpu")).to_s.downcase
    target_device = case requested_device
                    when "cpu"
                      MLX::Core.cpu
                    when "gpu", "metal"
                      if MLX::Core.respond_to?(:metal_is_available) && MLX::Core.metal_is_available
                        MLX::Core.gpu
                      else
                        MLX::Core.cpu
                      end
                    else
                      MLX::Core.cpu
                    end

    MLX::Core.set_default_device(target_device)
  rescue StandardError
    nil
  end
end

require_relative "mlx/extension"
require_relative "mlx/utils"
require_relative "mlx/nn"
require_relative "mlx/optimizers"
require_relative "mlx/dsl"

require_relative "mlx/distributed_utils/common"
require_relative "mlx/distributed_utils/config"
require_relative "mlx/distributed_utils/launch"
