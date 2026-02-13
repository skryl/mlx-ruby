# frozen_string_literal: true

module MLX
  module NN
    class << self
      def upsample_nearest(x, scale_factor)
        dims = x.ndim - 2
        if dims != scale_factor.length
          raise ArgumentError, "A scale needs to be provided for each spatial dimension"
        end

        result = x
        x.shape[1...-1].zip(scale_factor).each_with_index do |(n, scale), i|
          indices = nearest_indices(n, scale)
          result = MLX::Core.take(result, indices, 1 + i)
        end
        result
      end

      def upsample_linear(x, scale_factor, align_corners: false)
        _ = align_corners
        upsample_nearest(x, scale_factor)
      end

      def upsample_cubic(x, scale_factor, align_corners: false)
        _ = align_corners
        upsample_nearest(x, scale_factor)
      end

      private

      def nearest_indices(n, scale)
        m = (scale * n).to_i
        if m <= 0
          raise ArgumentError, "scale_factor must produce a positive output size"
        end

        if m > n
          indices = Array.new(m) do |i|
            (((i + 0.5) * (n.to_f / m)) - 0.5).round
          end
        else
          indices = Array.new(m) do |i|
            (i * (n.to_f / m)).floor
          end
        end
        indices.map { |idx| [[idx, 0].max, n - 1].min }
      end
    end

    class Upsample < Module
      def initialize(scale_factor:, mode: "nearest", align_corners: false)
        super()
        unless %w[nearest linear cubic].include?(mode)
          raise ArgumentError, "[Upsample] Got unsupported upsampling algorithm: #{mode}"
        end

        @scale_factor = if scale_factor.is_a?(Array)
          scale_factor.map(&:to_f)
        else
          scale_factor.to_f
        end
        @mode = mode
        @align_corners = align_corners
      end

      def call(x)
        dims = x.ndim - 2
        if dims <= 0
          raise ArgumentError,
                "[Upsample] The input should have at least 1 spatial dimension which means it should be at least 3D but #{x.ndim}D was provided"
        end

        scale = @scale_factor
        if scale.is_a?(Array)
          if scale.length != dims
            raise ArgumentError,
                  "[Upsample] One scale per spatial dimension is required but scale_factor=#{scale} and the number of spatial dimensions were #{dims}"
          end
        else
          scale = Array.new(dims, scale)
        end

        case @mode
        when "nearest"
          MLX::NN.upsample_nearest(x, scale)
        when "linear"
          MLX::NN.upsample_linear(x, scale, align_corners: @align_corners)
        when "cubic"
          MLX::NN.upsample_cubic(x, scale, align_corners: @align_corners)
        else
          raise ArgumentError, "Unknown interpolation mode: #{@mode}"
        end
      end
    end
  end
end
