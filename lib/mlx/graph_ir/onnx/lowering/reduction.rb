# frozen_string_literal: true

module MLX
  module GraphIR
    module ONNX
      module Lowering
        module Reduction
          OPS = %w[Reduce LogSumExp Scan ArgReduce].freeze
          module_function

          def handles?(op)
            OPS.include?(op)
          end

          def lower(graph_ir:, node:, node_index:, initializers:, used_tensor_names:, known_shapes:, known_dtypes:)
            graph_ir.send(
              :lower_onnx_node_default,
              node,
              node_index,
              initializers,
              used_tensor_names,
              known_shapes,
              known_dtypes
            )
          end
        end
      end
    end
  end
end
