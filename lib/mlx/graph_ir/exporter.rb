# frozen_string_literal: true

module MLX
  module GraphIR
    module Exporter
      module_function

      def export_graph_ir_json(fun, *extras, **trace_kwargs)
        MLX::Core.ensure_native!
        native = graph_ir_native_module
        unless native.respond_to?(:export_graph_ir_json)
          raise RuntimeError,
                "MLX::GraphIR::Native.export_graph_ir_json is unavailable; rebuild ext/mlx to a compatible native ABI"
        end

        args = extras.dup
        args << trace_kwargs unless trace_kwargs.empty?
        content = native.export_graph_ir_json(fun, *args)
        unless content.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.export_graph_ir_json must return String JSON"
        end
        content
      end

      def graph_ir_native_module
        unless defined?(MLX::GraphIR::Native)
          raise RuntimeError,
                "MLX::GraphIR::Native is unavailable; rebuild ext/mlx to a compatible native ABI"
        end

        MLX::GraphIR::Native
      end
      private_class_method :graph_ir_native_module
    end
  end
end
