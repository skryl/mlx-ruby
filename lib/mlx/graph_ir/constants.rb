# frozen_string_literal: true

module MLX
  module GraphIR
    IR_VERSION = 1

    REQUIRED_TOP_LEVEL_KEYS = %w[
      ir_version
      shapeless
      inputs
      keyword_inputs
      outputs
      constants
      nodes
    ].freeze

    SUPPORTED_DTYPES = %w[
      bool
      bool_
      uint8
      uint16
      uint32
      uint64
      int8
      int16
      int32
      int64
      float16
      float32
      float64
      bfloat16
      complex64
    ].freeze

    ONNX_OP_MAP = {
      "Add" => "Add",
      "AddMM" => "Gemm",
      "Subtract" => "Sub",
      "Multiply" => "Mul",
      "Square" => "Mul",
      "Divide" => "Div",
      "AsType" => "Cast",
      "Exp" => "Exp",
      "Log" => "Log",
      "Sin" => "Sin",
      "Cos" => "Cos",
      "Erf" => "Erf",
      "Sqrt" => "Sqrt",
      "Abs" => "Abs",
      "Floor" => "Floor",
      "Negative" => "Neg",
      "Relu" => "Relu",
      "Sigmoid" => "Sigmoid",
      "Tanh" => "Tanh",
      "Softmax" => "Softmax",
      "Greater" => "Greater",
      "Less" => "Less",
      "Equal" => "Equal",
      "Select" => "Where",
      "Full" => "Identity",
      "Matmul" => "MatMul",
      "Reshape" => "Reshape",
      "Flatten" => "Reshape",
      "Unflatten" => "Reshape",
      "Transpose" => "Transpose",
      "Squeeze" => "Squeeze",
      "ExpandDims" => "Unsqueeze",
      "Broadcast" => "Expand",
      "Arange" => "Constant",
      "AsStrided" => "Gather",
      "Concatenate" => "Concat",
      "Convolution" => "Conv",
      "ConvolutionTranspose" => "ConvTranspose",
      "Gather" => "Gather",
      "GatherAxis" => "GatherElements",
      "Slice" => "Slice",
      "Split" => "Split",
      "LogSumExp" => "ReduceLogSumExp",
      "Pad" => "Pad",
      "Scan" => "CumSum",
      "ScatterAxis" => "ScatterElements",
      "Maximum" => "Max",
      "Minimum" => "Min",
      "Power" => "Pow"
    }.freeze

    REDUCE_CODE_TO_ONNX_OP = {
      0 => "ReduceMin",
      1 => "ReduceMax",
      2 => "ReduceSum",
      3 => "ReduceProd",
      4 => "ReduceMin",
      5 => "ReduceMax"
    }.freeze

    ARG_REDUCE_CODE_TO_ONNX_OP = {
      0 => "ArgMin",
      1 => "ArgMax"
    }.freeze

    ONNX_DTYPE_MAP = {
      "bool" => "BOOL",
      "bool_" => "BOOL",
      "uint8" => "UINT8",
      "uint16" => "UINT16",
      "uint32" => "UINT32",
      "uint64" => "UINT64",
      "int8" => "INT8",
      "int16" => "INT16",
      "int32" => "INT32",
      "int64" => "INT64",
      "float16" => "FLOAT16",
      "float32" => "FLOAT",
      "float64" => "DOUBLE",
      "bfloat16" => "BFLOAT16",
      "complex64" => "COMPLEX64"
    }.freeze

    DTYPE_PROMOTION_ORDER = %w[
      bool
      uint8
      int8
      uint16
      int16
      uint32
      int32
      uint64
      int64
      bfloat16
      float16
      float32
      float64
    ].freeze
    DTYPE_PROMOTION_RANK = DTYPE_PROMOTION_ORDER.each_with_index.each_with_object({}) do |(dtype, rank), out|
      out[dtype] = rank
    end.freeze
    FILE_READ_CHUNK_BYTES = 8 * 1024 * 1024
    GRAPH_IR_NORMALIZATION_MAX_BYTES = 2_000_000_000
  end
end
