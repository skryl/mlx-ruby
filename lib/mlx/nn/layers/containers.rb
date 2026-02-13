# frozen_string_literal: true

module MLX
  module NN
    class Sequential < Module
      def initialize(*modules)
        super()
        self.layers = modules
      end

      def call(x)
        layers.each do |layer|
          x = layer.call(x)
        end
        x
      end
    end

  end
end
