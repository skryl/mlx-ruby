# frozen_string_literal: true

module MLX
  module DSL
    def self.splits(&block)
      plan = SplitPlan.new
      plan.instance_eval(&block) if block_given?
      plan
    end

    class SplitPlan
      SPLIT_OPTION_KEYS = %i[collate transform limit reduce].freeze
      SPLIT_NAMES = %i[train validation test].freeze

      def initialize
        @shared_options = {}
        @splits = SPLIT_NAMES.each_with_object({}) do |name, out|
          out[name] = { dataset: nil, options: {} }
        end
      end

      def shared(**options)
        @shared_options.merge!(__dsl_normalize_split_options(options, split: :shared))
        self
      end

      SPLIT_NAMES.each do |name|
        define_method(name) do |dataset = nil, **options, &block|
          __dsl_set_split(name, dataset, options, &block)
        end
      end

      def to_fit_inputs
        train = __dsl_effective_split(:train)
        validation = __dsl_effective_split(:validation)
        raise ArgumentError, "split plan requires a train dataset" if train[:dataset].nil?

        fit_kwargs = {}
        fit_kwargs[:collate] = train[:options][:collate] if train[:options].key?(:collate)
        fit_kwargs[:train_transform] = train[:options][:transform] if train[:options].key?(:transform)
        fit_kwargs[:limit] = train[:options][:limit] if train[:options].key?(:limit)
        fit_kwargs[:reduce] = train[:options][:reduce] if train[:options].key?(:reduce)

        unless validation[:dataset].nil?
          fit_kwargs[:validation_data] = validation[:dataset]
          fit_kwargs[:validation_collate] = validation[:options][:collate] if validation[:options].key?(:collate)
          fit_kwargs[:validation_transform] = validation[:options][:transform] if validation[:options].key?(:transform)
          fit_kwargs[:validation_limit] = validation[:options][:limit] if validation[:options].key?(:limit)
          fit_kwargs[:validation_reduce] = validation[:options][:reduce] if validation[:options].key?(:reduce)
        end

        [train[:dataset], fit_kwargs]
      end

      private

      def __dsl_set_split(name, dataset, options, &block)
        resolved_dataset = block_given? ? block.call : dataset
        @splits[name][:dataset] = resolved_dataset unless resolved_dataset.nil?
        @splits[name][:options].merge!(__dsl_normalize_split_options(options, split: name))
        self
      end

      def __dsl_effective_split(name)
        split = @splits.fetch(name)
        {
          dataset: split[:dataset],
          options: @shared_options.merge(split[:options])
        }
      end

      def __dsl_normalize_split_options(options, split:)
        normalized = (options || {}).each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        unknown = normalized.keys - SPLIT_OPTION_KEYS
        unless unknown.empty?
          raise ArgumentError, "#{split} split has unsupported key(s): #{unknown.map(&:inspect).join(', ')}"
        end

        normalized
      end
    end
  end
end
