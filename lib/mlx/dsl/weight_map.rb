# frozen_string_literal: true

module MLX
  module DSL
    def self.weight_map(&block)
      mapper = WeightMap.new
      mapper.instance_eval(&block) if block_given?
      mapper
    end

    class WeightMap
      def initialize
        @rules = []
      end

      def strip_prefix(prefix)
        @rules << [:strip_prefix, prefix.to_s]
        self
      end

      def rename(from = nil, to = nil)
        if from.is_a?(Hash)
          from.each do |src, dst|
            @rules << [:rename, src.to_s, dst.to_s]
          end
          return self
        end

        if from.nil? || to.nil?
          raise ArgumentError, "rename requires either a mapping hash or from/to arguments"
        end

        @rules << [:rename, from.to_s, to.to_s]
        self
      end

      def regex(pattern, replacement = nil, &block)
        if pattern.nil?
          raise ArgumentError, "regex requires a Regexp pattern"
        end
        if replacement.nil? && !block_given?
          raise ArgumentError, "regex requires a replacement argument or block"
        end

        @rules << [:regex, pattern, replacement, block]
        self
      end

      def split_qkv(source, into:, axis: 0)
        names = into.to_a.map(&:to_s)
        if names.empty? || names.length < 2
          raise ArgumentError, "split_qkv :into must include at least two output names"
        end

        @rules << [:split, source.to_s, names, axis.to_i]
        self
      end

      def transpose_if(rank:, order:)
        @rules << [:transpose_if, rank.to_i, order.to_a]
        self
      end

      def apply(weights)
        entries = if weights.is_a?(Hash)
          weights.to_a
        elsif weights.respond_to?(:to_a)
          weights.to_a
        else
          raise ArgumentError, "weights must be a Hash or array-like key/value collection"
        end

        out = {}
        entries.each do |entry|
          key, value = entry
          __dsl_apply_rules_for_entry(key.to_s, value).each do |mapped_key, mapped_value|
            out[mapped_key] = mapped_value
          end
        end
        out
      end

      private

      def __dsl_apply_rules_for_entry(key, value)
        current_entries = [[key, value]]
        @rules.each do |rule|
          current_entries = current_entries.flat_map do |curr_key, curr_value|
            __dsl_apply_rule(rule, curr_key, curr_value)
          end
        end
        current_entries
      end

      def __dsl_apply_rule(rule, key, value)
        kind = rule[0]
        case kind
        when :strip_prefix
          prefix = rule[1]
          if key.start_with?(prefix)
            [[key[prefix.length..], value]]
          else
            [[key, value]]
          end
        when :rename
          from = rule[1]
          to = rule[2]
          [[key.gsub(from, to), value]]
        when :regex
          pattern = rule[1]
          replacement = rule[2]
          block = rule[3]
          if block.nil?
            [[key.gsub(pattern, replacement.to_s), value]]
          else
            [[key.gsub(pattern, &block), value]]
          end
        when :split
          source = rule[1]
          targets = rule[2]
          axis = rule[3]
          return [[key, value]] unless key == source

          parts = MLX::Core.split(value, targets.length, axis)
          targets.each_with_index.map { |target, index| [target, parts[index]] }
        when :transpose_if
          target_rank = rule[1]
          order = rule[2]
          if value.respond_to?(:shape) && value.shape.length == target_rank
            [[key, MLX::Core.transpose(value, order)]]
          else
            [[key, value]]
          end
        else
          [[key, value]]
        end
      end
    end
  end
end
