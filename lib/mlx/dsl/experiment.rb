# frozen_string_literal: true

module MLX
  module DSL
    def self.experiment(name = nil, &block)
      instance = Experiment.new(name: name)
      instance.instance_eval(&block) if block_given?
      instance
    end

    class Experiment
      attr_reader :name

      def initialize(name: nil)
        @name = name
        @model_source = nil
        @optimizer_source = nil
        @trainer_source = nil
        @trainer_kwargs = {}
        @loss_block = nil
        @data_config = { train: nil, validation: nil, fit: {} }
        @artifact_config = {}
        @last_trainer = nil
        @last_report = nil
      end

      def model(value = nil, &block)
        if !value.nil? && block_given?
          raise ArgumentError, "model accepts either a value argument or block, not both"
        end

        @model_source = block_given? ? block : value
        self
      end

      def optimizer(value = nil, &block)
        if !value.nil? && block_given?
          raise ArgumentError, "optimizer accepts either a value argument or block, not both"
        end

        @optimizer_source = block_given? ? block : value
        self
      end

      def trainer(value = nil, **kwargs, &block)
        if value.is_a?(MLX::DSL::Trainer)
          if !kwargs.empty? || block_given?
            raise ArgumentError, "trainer instance injection cannot be combined with trainer kwargs or loss block"
          end
          @trainer_source = value
          return self
        end

        unless value.nil?
          raise ArgumentError, "trainer positional argument must be an MLX::DSL::Trainer instance"
        end

        @trainer_source = nil
        @trainer_kwargs = kwargs.dup
        @loss_block = block if block_given?
        self
      end

      def data(train: nil, validation: :__dsl_unset__, **fit_kwargs)
        @data_config[:train] = train unless train.nil?
        @data_config[:validation] = validation unless validation == :__dsl_unset__
        @data_config[:fit].merge!(fit_kwargs)
        self
      end

      def artifacts(**kwargs)
        @artifact_config.merge!(kwargs)
        self
      end

      def run(report: false, **overrides)
        dataset, fit_kwargs = __dsl_resolve_fit_call(overrides)
        active_trainer = __dsl_resolve_trainer
        result = if report
          active_trainer.fit_report(dataset, **fit_kwargs)
        else
          active_trainer.fit(dataset, **fit_kwargs)
        end
        @last_report = result if report
        result
      end

      def report(**overrides)
        run(report: true, **overrides)
      end

      def save_run_bundle(path, report: nil, config: {}, **overrides)
        active_report = report
        if active_report.nil?
          active_report = if !@last_report.nil?
            @last_report
          else
            self.report(**overrides)
          end
        end

        __dsl_resolve_trainer.save_run_bundle(path, report: active_report, config: config)
      end

      private

      def __dsl_resolve_fit_call(overrides)
        fit_kwargs = @data_config.fetch(:fit).dup
        fit_kwargs.merge!(@artifact_config)
        fit_kwargs[:validation_data] = @data_config[:validation] if !@data_config[:validation].nil? && !fit_kwargs.key?(:validation_data)

        incoming = overrides.dup
        dataset = if incoming.key?(:dataset)
          incoming.delete(:dataset)
        elsif incoming.key?(:train)
          incoming.delete(:train)
        else
          @data_config[:train]
        end
        if dataset.nil?
          raise ArgumentError, "experiment run requires a train dataset via data(train:) or run(dataset:)"
        end

        [dataset, fit_kwargs.merge(incoming)]
      end

      def __dsl_resolve_trainer
        return @trainer_source if @trainer_source.is_a?(MLX::DSL::Trainer)
        return @last_trainer unless @last_trainer.nil?

        model = __dsl_resolve_source(@model_source, "model")
        optimizer = __dsl_resolve_source(@optimizer_source, "optimizer")
        unless model.respond_to?(:trainer)
          raise ArgumentError, "experiment model must respond to #trainer when trainer instance is not injected"
        end
        unless @loss_block.respond_to?(:call)
          raise ArgumentError, "experiment trainer requires a loss block when trainer instance is not injected"
        end

        @last_trainer = model.trainer(optimizer: optimizer, **@trainer_kwargs, &@loss_block)
      end

      def __dsl_resolve_source(source, label)
        value = source
        value = value.call if value.respond_to?(:call)
        if value.nil?
          raise ArgumentError, "experiment #{label} section is required when trainer instance is not injected"
        end

        value
      end
    end
  end
end
