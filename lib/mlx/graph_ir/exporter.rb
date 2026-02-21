# frozen_string_literal: true

module MLX
  module GraphIR
    module Exporter
      module_function

      def export_graph_ir_json(fun, *extras, **trace_kwargs)
        MLX::Core.ensure_native!
        args = extras.dup
        args << trace_kwargs unless trace_kwargs.empty?
        content = MLX::GraphIR::Native.export_graph_ir_json(fun, *args)
        unless content.is_a?(String)
          raise TypeError, "MLX::GraphIR::Native.export_graph_ir_json must return String JSON"
        end
        content
      end
    end
  end
end
