# frozen_string_literal: true

module MLX
  module DSL
    class Trainer
      HOOK_EVENTS = %i[
        before_fit
        before_epoch
        after_batch
        before_validation
        after_validation_batch
        after_validation
        after_epoch
        checkpoint
        after_fit
      ].freeze

      def initialize(model:, optimizer:, clip_grad_norm: nil, &loss_block)
        raise ArgumentError, "trainer requires a loss block" unless block_given?

        @model = model
        @loss_block = loss_block
        @step = model.train_step(
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
          &loss_block
        )
        @optimizer = optimizer
        @hooks = Hash.new { |h, k| h[k] = [] }
      end

      def on(event, &block)
        raise ArgumentError, "hook registration requires a block" unless block_given?

        @hooks[event.to_sym] << block
        self
      end

      HOOK_EVENTS.each do |event|
        define_method(event) do |&block|
          on(event, &block)
        end
      end

      def fit(
        dataset,
        epochs: 1,
        limit: nil,
        report: false,
        reduce: :mean,
        monitor: :epoch_loss,
        metric: nil,
        validation_data: nil,
        validation_reduce: nil,
        train_transform: nil,
        validation_transform: nil,
        checkpoint_path: nil,
        save_best: false,
        monitor_mode: :min,
        patience: nil,
        min_delta: 0.0,
        keep_losses: true,
        strict_data_reuse: false,
        metadata: {}
      )
        keep_losses = !!keep_losses
        strict_data_reuse = !!strict_data_reuse
        losses = []
        epoch_rows = []
        best_metric = nil
        stale_epochs = 0
        stopped_early = false
        previous_train_batches = nil
        previous_validation_batches = nil
        monitor_name = monitor.to_s
        patience_value = __dsl_normalize_patience(patience)
        min_delta_value = __dsl_normalize_min_delta(min_delta)
        train_dataset_size = __dsl_dataset_size(dataset)
        validation_dataset_size = __dsl_dataset_size(validation_data)
        validation_reducer = validation_reduce.nil? ? reduce : validation_reduce

        emit(
          :before_fit,
          {
            model: @model,
            optimizer: @optimizer,
            epochs: epochs.to_i,
            dataset_size: train_dataset_size,
            validation_size: validation_dataset_size
          }
        )

        epochs.to_i.times do |epoch|
          emit(:before_epoch, { epoch: epoch, model: @model })
          index = 0
          epoch_losses = []
          __dsl_dataset_for_epoch(dataset, epoch: epoch, kind: :train).each do |batch|
            break if !limit.nil? && index >= limit

            batch = __dsl_apply_batch_transform(
              train_transform,
              batch,
              epoch: epoch,
              batch_index: index,
              kind: :train
            )
            loss = __dsl_run_batch(batch)
            losses << loss if keep_losses
            scalar = __dsl_loss_scalar(loss)
            epoch_losses << scalar unless scalar.nil?
            emit(
              :after_batch,
              {
                epoch: epoch,
                batch_index: index,
                loss: loss,
                loss_value: scalar,
                model: @model
              }
            )
            index += 1
          end
          __dsl_validate_data_reuse!(
            strict: strict_data_reuse,
            dataset: dataset,
            kind: :train,
            epoch: epoch,
            previous_batches: previous_train_batches,
            current_batches: index
          )
          previous_train_batches = index

          epoch_metric = __dsl_reduce_values(epoch_losses, reduce)
          validation_losses = []
          validation_batch_count = 0
          val_metric = nil
          unless validation_data.nil?
            emit(
              :before_validation,
              {
                epoch: epoch,
                model: @model,
                monitor_name: monitor_name
              }
            )
            __dsl_with_eval_mode do
              __dsl_dataset_for_epoch(validation_data, epoch: epoch, kind: :validation).each do |batch|
                batch = __dsl_apply_batch_transform(
                  validation_transform,
                  batch,
                  epoch: epoch,
                  batch_index: validation_batch_count,
                  kind: :validation
                )
                loss = __dsl_run_validation_batch(batch)
                scalar = __dsl_loss_scalar(loss)
                validation_losses << scalar unless scalar.nil?
                emit(
                  :after_validation_batch,
                  {
                    epoch: epoch,
                    batch_index: validation_batch_count,
                    loss: loss,
                    loss_value: scalar,
                    model: @model
                  }
                )
                validation_batch_count += 1
              end
            end
            __dsl_validate_data_reuse!(
              strict: strict_data_reuse,
              dataset: validation_data,
              kind: :validation,
              epoch: epoch,
              previous_batches: previous_validation_batches,
              current_batches: validation_batch_count
            )
            previous_validation_batches = validation_batch_count
            val_metric = __dsl_reduce_values(validation_losses, validation_reducer)
            emit(
              :after_validation,
              {
                epoch: epoch,
                model: @model,
                val_loss: val_metric,
                validation_batches: validation_batch_count
              }
            )
          end

          monitor_value = __dsl_monitor_value(
            metric,
            {
              epoch: epoch,
              epoch_losses: epoch_losses,
              epoch_loss: epoch_metric,
              val_loss: val_metric,
              validation_losses: validation_losses,
              losses: losses,
              batches: index,
              validation_batches: validation_batch_count,
              model: @model,
              optimizer: @optimizer
            },
            fallback: __dsl_default_monitor_value(monitor_name, epoch_metric, val_metric)
          )
          improved = __dsl_improved?(
            monitor_value,
            best_metric,
            monitor_mode,
            min_delta: min_delta_value
          )
          if improved
            best_metric = monitor_value
            stale_epochs = 0
          elsif !best_metric.nil?
            stale_epochs += 1
          end

          row = {
            "epoch" => epoch,
            "batches" => index,
            "epoch_loss" => epoch_metric,
            "val_loss" => val_metric,
            "validation_batches" => validation_batch_count,
            "monitor_value" => monitor_value,
            "stale_epochs" => stale_epochs,
            "improved" => improved
          }
          epoch_rows << row

          checkpoint_saved = __dsl_maybe_checkpoint(
            checkpoint_path,
            save_best: save_best,
            improved: improved,
            epoch: epoch,
            monitor_name: monitor_name,
            monitor_value: monitor_value,
            epoch_metric: epoch_metric,
            metadata: metadata
          )

          emit(
            :after_epoch,
            {
              epoch: epoch,
              model: @model,
              epoch_loss: epoch_metric,
              val_loss: val_metric,
              monitor_name: monitor_name,
              monitor_value: monitor_value,
              validation_batches: validation_batch_count,
              stale_epochs: stale_epochs,
              improved: improved,
              best_metric: best_metric,
              checkpoint_saved: checkpoint_saved
            }
          )

          if !patience_value.nil? && stale_epochs > patience_value
            stopped_early = true
            break
          end
        end

        payload = {
          "losses" => losses,
          "losses_kept" => keep_losses,
          "epochs" => epoch_rows,
          "monitor_name" => monitor_name,
          "epochs_ran" => epoch_rows.length,
          "stopped_early" => stopped_early,
          "best_metric" => best_metric
        }
        emit(
          :after_fit,
          {
            model: @model,
            optimizer: @optimizer,
            epochs: epoch_rows.length,
            best_metric: best_metric,
            stopped_early: stopped_early,
            report: payload
          }
        )

        return payload if report

        losses
      end

      def fit_report(dataset, **kwargs)
        fit(dataset, **kwargs, report: true)
      end

      private

      def emit(event, context)
        @hooks[event.to_sym].each do |hook|
          hook.call(context)
        end
      end

      def __dsl_run_batch(batch)
        if batch.is_a?(Hash)
          @step.call(**batch)
        elsif batch.is_a?(Array)
          @step.call(*batch)
        else
          @step.call(batch)
        end
      end

      def __dsl_loss_scalar(loss)
        return nil if loss.nil?
        return loss.to_f if loss.is_a?(Numeric)

        return loss.item.to_f if loss.respond_to?(:item)
        return loss.to_f if loss.respond_to?(:to_f)

        nil
      end

      def __dsl_default_monitor_value(monitor_name, epoch_metric, val_metric)
        case monitor_name.to_s
        when "val_loss", "validation_loss"
          val_metric.nil? ? epoch_metric : val_metric
        else
          epoch_metric
        end
      end

      def __dsl_monitor_value(metric, context, fallback:)
        return fallback unless metric.respond_to?(:call)

        metric.call(context)
      end

      def __dsl_run_validation_batch(batch)
        if batch.is_a?(Hash)
          @loss_block.call(**batch)
        elsif batch.is_a?(Array)
          @loss_block.call(*batch)
        else
          @loss_block.call(batch)
        end
      end

      def __dsl_reduce_values(values, reducer)
        return nil if values.empty?

        if reducer.respond_to?(:call)
          return reducer.call(values)
        end

        case reducer.to_sym
        when :mean
          values.sum / values.length.to_f
        when :sum
          values.sum
        when :last
          values[-1]
        else
          raise ArgumentError, "unsupported reducer: #{reducer.inspect}"
        end
      end

      def __dsl_improved?(metric, best_metric, monitor_mode, min_delta: 0.0)
        return false if metric.nil?
        return true if best_metric.nil?

        case monitor_mode.to_sym
        when :min
          metric < (best_metric - min_delta)
        when :max
          metric > (best_metric + min_delta)
        else
          raise ArgumentError, "unsupported monitor_mode: #{monitor_mode.inspect}"
        end
      end

      def __dsl_normalize_patience(patience)
        return nil if patience.nil?

        value = patience.to_i
        raise ArgumentError, "patience must be non-negative" if value.negative?

        value
      end

      def __dsl_apply_batch_transform(transform, batch, epoch:, batch_index:, kind:)
        return batch if transform.nil?
        unless transform.respond_to?(:call)
          raise ArgumentError, "#{kind} transform must respond to #call"
        end

        values = {
          batch: batch,
          epoch: epoch,
          batch_index: batch_index,
          kind: kind,
          trainer: self
        }
        return transform.call(batch) unless transform.respond_to?(:parameters)

        params = transform.parameters
        return transform.call(batch) if params.empty?

        args = __dsl_build_positional_args(
          params,
          values,
          [[:batch, batch], [:epoch, epoch], [:batch_index, batch_index], [:kind, kind], [:trainer, self]],
          "batch transform"
        )
        kwargs = __dsl_build_keyword_args(params, values, "batch transform")
        return transform.call(*args) if kwargs.empty?

        transform.call(*args, **kwargs)
      end

      def __dsl_normalize_min_delta(min_delta)
        value = min_delta.to_f
        raise ArgumentError, "min_delta must be non-negative" if value.negative?

        value
      end

      def __dsl_maybe_checkpoint(
        path,
        save_best:,
        improved:,
        epoch:,
        monitor_name:,
        monitor_value:,
        epoch_metric:,
        metadata:
      )
        return false if path.nil? || path.to_s.empty?
        return false if save_best && !improved
        return false unless @model.respond_to?(:save_checkpoint)

        resolved_path = __dsl_checkpoint_path(
          path,
          epoch: epoch,
          monitor_name: monitor_name,
          monitor_value: monitor_value,
          epoch_metric: epoch_metric,
          improved: improved
        )

        merged_metadata = (metadata || {}).dup
        merged_metadata["epoch"] = epoch
        merged_metadata["epoch_loss"] = epoch_metric
        merged_metadata["monitor_name"] = monitor_name
        merged_metadata["monitor_value"] = monitor_value

        @model.save_checkpoint(resolved_path, optimizer: @optimizer, metadata: merged_metadata)
        emit(
          :checkpoint,
          {
            model: @model,
            optimizer: @optimizer,
            path: resolved_path,
            epoch: epoch,
            monitor_name: monitor_name,
            monitor_value: monitor_value,
            epoch_loss: epoch_metric,
            improved: improved
          }
        )
        true
      end

      def __dsl_checkpoint_path(path, epoch:, monitor_name:, monitor_value:, epoch_metric:, improved:)
        return path unless path.include?("%{")

        path % {
          epoch: epoch,
          monitor: monitor_value,
          monitor_name: monitor_name,
          epoch_loss: epoch_metric,
          improved: improved
        }
      rescue KeyError => e
        raise ArgumentError, "unsupported checkpoint template key: #{e.message}"
      end

      def __dsl_dataset_size(dataset)
        return nil if dataset.nil? || dataset.respond_to?(:call)
        return dataset.size if dataset.respond_to?(:size)
        return dataset.length if dataset.respond_to?(:length)

        nil
      end

      def __dsl_dataset_for_epoch(dataset, epoch:, kind:)
        source = if dataset.respond_to?(:call)
          __dsl_call_dataset_factory(dataset, epoch: epoch, kind: kind)
        else
          dataset
        end
        unless source.respond_to?(:each)
          raise ArgumentError, "#{kind} dataset must respond to #each"
        end

        if epoch.positive? && !dataset.respond_to?(:call) && source.respond_to?(:rewind)
          source.rewind
        end

        source
      rescue StandardError => e
        raise ArgumentError, "#{kind} dataset could not rewind for epoch #{epoch}: #{e.message}"
      end

      def __dsl_call_dataset_factory(factory, epoch:, kind:)
        return factory.call unless factory.respond_to?(:parameters)

        params = factory.parameters
        values = {
          epoch: epoch,
          trainer: self,
          kind: kind
        }
        return factory.call(epoch) if params.empty?

        args = __dsl_build_positional_args(
          params,
          values,
          [[:epoch, epoch], [:kind, kind], [:trainer, self]],
          "dataset factory"
        )
        kwargs = __dsl_build_keyword_args(params, values, "dataset factory")
        return factory.call(*args) if kwargs.empty?

        factory.call(*args, **kwargs)
      end

      def __dsl_build_positional_args(params, values, fallback_pairs, label)
        queue = fallback_pairs.dup
        args = []
        params.each do |type, name|
          next unless type == :req || type == :opt

          if !name.nil? && values.key?(name)
            args << values.fetch(name)
            queue.reject! { |key, _value| key == name }
            next
          end

          if queue.empty?
            raise ArgumentError, "#{label} has unsupported required positional argument: #{name.inspect}" if type == :req
            break
          end

          _key, value = queue.shift
          args << value
        end
        args
      end

      def __dsl_build_keyword_args(params, values, label)
        return values.dup if params.any? { |type, _name| type == :keyrest }

        required_keys = params.each_with_object([]) do |(type, name), out|
          out << name if type == :keyreq
        end
        missing = required_keys.reject { |name| values.key?(name) }
        unless missing.empty?
          raise ArgumentError, "#{label} requires unsupported keyword argument(s): #{missing.map(&:inspect).join(", ")}"
        end

        accepted_keys = params.each_with_object([]) do |(type, name), out|
          out << name if type == :key || type == :keyreq
        end

        values.each_with_object({}) do |(name, value), out|
          out[name] = value if accepted_keys.include?(name)
        end
      end

      def __dsl_validate_data_reuse!(strict:, dataset:, kind:, epoch:, previous_batches:, current_batches:)
        return unless strict
        return if dataset.nil? || dataset.respond_to?(:call)
        return unless epoch.positive?
        return unless previous_batches.to_i.positive? && current_batches.to_i.zero?
        return if dataset.respond_to?(:rewind)

        raise ArgumentError,
              "#{kind} dataset appears exhausted across epochs; pass a factory like ->(epoch:) { ... }"
      end

      def __dsl_with_eval_mode
        if @model.respond_to?(:eval_mode)
          @model.eval_mode { yield }
          return
        end

        unless @model.respond_to?(:eval) && @model.respond_to?(:train) && @model.respond_to?(:training)
          yield
          return
        end

        previous = @model.training
        @model.eval
        yield
      ensure
        @model.train(previous) unless previous.nil?
      end
    end
  end
end
