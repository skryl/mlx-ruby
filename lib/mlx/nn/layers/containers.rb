# frozen_string_literal: true

module MLX
  module NN
    class Sequential < Module
      def initialize(*modules)
        super()
        self.layers = modules
      end

      def call(*args, **kwargs)
        out = nil
        layers.each_with_index do |layer, index|
          out = if index.zero?
            kwargs.empty? ? layer.call(*args) : layer.call(*args, **kwargs)
          else
            __dsl_forward_layer_output(layer, out)
          end
        end
        out
      end

      private

      def __dsl_forward_layer_output(layer, out)
        if out.is_a?(::Array)
          layer.call(*out)
        elsif out.is_a?(::Hash)
          layer.call(**out)
        else
          layer.call(out)
        end
      end
    end

  end
end
