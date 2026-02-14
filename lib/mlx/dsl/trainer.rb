# frozen_string_literal: true

require "json"
require "fileutils"
require "time"

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
      UNSET = Object.new.freeze
      FIT_OPTION_DEFAULTS = {
        epochs: 1,
        limit: nil,
        collate: nil,
        bind: nil,
        report: false,
        reduce: :mean,
        monitor: :epoch_loss,
        metric: nil,
        validation_data: nil,
        validation_limit: nil,
        validation_reduce: nil,
        validation_collate: nil,
        validation_bind: nil,
        train_transform: nil,
        validation_transform: nil,
        checkpoint_path: nil,
        save_best: false,
        monitor_mode: :min,
        patience: nil,
        min_delta: 0.0,
        keep_losses: true,
        strict_data_reuse: false,
        resume_from: nil,
        metadata: {}
      }.freeze

      def initialize(model:, optimizer:, clip_grad_norm: nil, compile: false, sync: :none, &loss_block)
        raise ArgumentError, "trainer requires a loss block" unless block_given?

        @__dsl_init_options = {
          model: model,
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
          compile: compile,
          sync: sync
        }
        @model = model
        @loss_block = loss_block
        @sync_mode = __dsl_normalize_sync(sync)
        @step = __dsl_build_train_step(
          model,
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
          compile: compile,
          &loss_block
        )
        @optimizer = optimizer
        @hooks = Hash.new { |h, k| h[k] = [] }
        @hook_order = 0
        @collate_registry = {}
        @fit_defaults = {}
        @fit_presets = {}
        @batch_schemas = { train: nil, validation: nil }
        @dataflow_profiles = {}
        @hook_packs = {}
        @metric_registry = {}
        @task_presets = __dsl_builtin_task_presets
        @artifact_policy_config = {
          checkpoint: {},
          retention: {},
          resume: nil,
          run_bundle: {}
        }
        @checkpoint_history = []
        @last_checkpoint_snapshot = nil
      end

      def on(event, priority: 0, every: nil, once: false, **kwargs, &block)
        raise ArgumentError, "hook registration requires a block" unless block_given?
        condition = kwargs.delete(:if)
        condition = kwargs.delete(:condition) if condition.nil? && kwargs.key?(:condition)
        unless kwargs.empty?
          raise ArgumentError, "unsupported hook option(s): #{kwargs.keys.map(&:inspect).join(', ')}"
        end
        every_value = nil
        unless every.nil?
          every_value = every.to_i
          raise ArgumentError, "hook :every must be a positive integer" if every_value <= 0
        end

        @hooks[event.to_sym] << {
          hook: block,
          priority: priority.to_i,
          every: every_value,
          once: !!once,
          if: condition,
          fired: false,
          invocations: 0,
          order: @hook_order
        }
        @hook_order += 1
        self
      end

      HOOK_EVENTS.each do |event|
        define_method(event) do |**options, &block|
          on(event, **options, &block)
        end
      end

      def fit(
        dataset,
        epochs: UNSET,
        limit: UNSET,
        collate: UNSET,
        bind: UNSET,
        report: UNSET,
        reduce: UNSET,
        monitor: UNSET,
        metric: UNSET,
        validation_data: UNSET,
        validation_limit: UNSET,
        validation_reduce: UNSET,
        validation_collate: UNSET,
        validation_bind: UNSET,
        train_transform: UNSET,
        validation_transform: UNSET,
        checkpoint_path: UNSET,
        save_best: UNSET,
        monitor_mode: UNSET,
        patience: UNSET,
        min_delta: UNSET,
        keep_losses: UNSET,
        strict_data_reuse: UNSET,
        resume_from: UNSET,
        metadata: UNSET
      )
        raw_options = {
          epochs: epochs,
          limit: limit,
          collate: collate,
          bind: bind,
          report: report,
          reduce: reduce,
          monitor: monitor,
          metric: metric,
          validation_data: validation_data,
          validation_limit: validation_limit,
          validation_reduce: validation_reduce,
          validation_collate: validation_collate,
          validation_bind: validation_bind,
          train_transform: train_transform,
          validation_transform: validation_transform,
          checkpoint_path: checkpoint_path,
          save_best: save_best,
          monitor_mode: monitor_mode,
          patience: patience,
          min_delta: min_delta,
          keep_losses: keep_losses,
          strict_data_reuse: strict_data_reuse,
          resume_from: resume_from,
          metadata: metadata
        }
        dataset, raw_options = __dsl_expand_split_plan(dataset, raw_options)
        options = __dsl_resolve_fit_options(raw_options)
        epochs = options.fetch(:epochs)
        limit = options.fetch(:limit)
        collate = options.fetch(:collate)
        bind = options.fetch(:bind)
        report = options.fetch(:report)
        reduce = options.fetch(:reduce)
        monitor = options.fetch(:monitor)
        metric = options.fetch(:metric)
        validation_data = options.fetch(:validation_data)
        validation_limit = options.fetch(:validation_limit)
        validation_reduce = options.fetch(:validation_reduce)
        validation_collate = options.fetch(:validation_collate)
        validation_bind = options.fetch(:validation_bind)
        train_transform = options.fetch(:train_transform)
        validation_transform = options.fetch(:validation_transform)
        checkpoint_path = options.fetch(:checkpoint_path)
        save_best = options.fetch(:save_best)
        monitor_mode = options.fetch(:monitor_mode)
        patience = options.fetch(:patience)
        min_delta = options.fetch(:min_delta)
        keep_losses = options.fetch(:keep_losses)
        strict_data_reuse = options.fetch(:strict_data_reuse)
        resume_from = options.fetch(:resume_from)
        metadata = options.fetch(:metadata)
        policy = __dsl_resolve_artifact_policy(
          checkpoint_path: checkpoint_path,
          save_best: save_best,
          resume_from: resume_from,
          monitor_mode: monitor_mode
        )
        checkpoint_path = policy.fetch(:checkpoint_path)
        save_best = policy.fetch(:save_best)
        resume_from = policy.fetch(:resume_from)
        checkpoint_every = policy.fetch(:checkpoint_every)
        retention_keep_last_n = policy.fetch(:keep_last_n)
        run_bundle_policy = policy.fetch(:run_bundle)
        policy_payload = policy.fetch(:payload)

        keep_losses = !!keep_losses
        strict_data_reuse = !!strict_data_reuse
        @last_checkpoint_snapshot = nil
        losses = []
        epoch_rows = []
        best_metric = nil
        stale_epochs = 0
        stopped_early = false
        previous_train_batches = nil
        previous_validation_batches = nil
        monitor_name = monitor.to_s
        resume_state = __dsl_resume_state(resume_from, monitor_name)
        start_epoch = resume_state.fetch(:start_epoch)
        best_metric = resume_state.fetch(:best_metric)
        stale_epochs = resume_state.fetch(:stale_epochs)
        monitor_name = resume_state.fetch(:monitor_name)
        total_epochs = epochs.to_i
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
            epochs: total_epochs,
            start_epoch: start_epoch,
            resumed_from_epoch: resume_state.fetch(:checkpoint_epoch),
            resume_from: resume_state.fetch(:path),
            best_metric: best_metric,
            stale_epochs: stale_epochs,
            dataset_size: train_dataset_size,
            validation_size: validation_dataset_size
          }
        )

        (start_epoch...total_epochs).each do |epoch|
          emit(:before_epoch, { epoch: epoch, model: @model })
          index = 0
          epoch_losses = []
          epoch_last_loss = nil
          train_limit = __dsl_resolve_loop_limit(limit, epoch: epoch, kind: :train)
          __dsl_dataset_for_epoch(dataset, epoch: epoch, kind: :train).each do |batch|
            break if !train_limit.nil? && index >= train_limit

            batch = __dsl_apply_batch_transform(
              train_transform,
              __dsl_apply_collate(
                __dsl_effective_collate(
                  collate,
                  bind,
                  batch,
                  kind: :train
                ),
                batch,
                kind: :train,
                epoch: epoch,
                batch_index: index
              ),
              epoch: epoch,
              batch_index: index,
              kind: :train
            )
            loss = __dsl_run_batch(
              batch,
              epoch: epoch,
              batch_index: index,
              kind: :train
            )
            epoch_last_loss = loss
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
            validation_epoch_limit = __dsl_resolve_loop_limit(
              validation_limit,
              epoch: epoch,
              kind: :validation
            )
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
                break if !validation_epoch_limit.nil? && validation_batch_count >= validation_epoch_limit

                batch = __dsl_apply_batch_transform(
                  validation_transform,
                  __dsl_apply_collate(
                    __dsl_effective_collate(
                      validation_collate,
                      validation_bind,
                      batch,
                      kind: :validation
                    ),
                    batch,
                    kind: :validation,
                    epoch: epoch,
                    batch_index: validation_batch_count
                  ),
                  epoch: epoch,
                  batch_index: validation_batch_count,
                  kind: :validation
                )
                loss = __dsl_run_validation_batch(
                  batch,
                  epoch: epoch,
                  batch_index: validation_batch_count,
                  kind: :validation
                )
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
            stale_epochs: stale_epochs,
            best_metric: best_metric,
            metadata: metadata,
            checkpoint_every: checkpoint_every,
            keep_last_n: retention_keep_last_n
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

          __dsl_sync_epoch(epoch_last_loss) if @sync_mode == :epoch

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
          "epochs_target" => total_epochs,
          "epochs_completed" => [start_epoch + epoch_rows.length, total_epochs].min,
          "epochs_ran" => epoch_rows.length,
          "stopped_early" => stopped_early,
          "best_metric" => best_metric,
          "resume_from" => resume_state.fetch(:path),
          "resumed_from_epoch" => resume_state.fetch(:checkpoint_epoch),
          "start_epoch" => start_epoch,
          "artifact_policy" => policy_payload
        }
        auto_bundle_path = __dsl_auto_save_run_bundle(
          run_bundle_policy,
          payload
        )
        payload["run_bundle_path"] = auto_bundle_path unless auto_bundle_path.nil?
        emit(
          :after_fit,
          {
            model: @model,
            optimizer: @optimizer,
            epochs: epoch_rows.length,
            best_metric: best_metric,
            stopped_early: stopped_early,
            resume_from: resume_state.fetch(:path),
            resumed_from_epoch: resume_state.fetch(:checkpoint_epoch),
            report: payload
          }
        )

        return payload if report

        losses
      end

      def fit_report(dataset, **kwargs)
        fit(dataset, **kwargs, report: true)
      end

      def with_fit_defaults(**defaults)
        configured = __dsl_clone_trainer
        configured.instance_variable_set(
          :@fit_defaults,
          configured.instance_variable_get(:@fit_defaults).merge(__dsl_normalize_fit_option_keys(defaults))
        )
        configured
      end

      def register_fit_preset(name, **defaults)
        @fit_presets[name.to_sym] = __dsl_normalize_fit_option_keys(defaults)
        self
      end

      def fit_with(name, dataset, **overrides)
        fit(dataset, **__dsl_merge_fit_preset(name, overrides))
      end

      def fit_report_with(name, dataset, **overrides)
        fit_report(dataset, **__dsl_merge_fit_preset(name, overrides))
      end

      def register_dataflow(name, train: {}, validation: {}, extends: nil)
        profile = __dsl_normalize_dataflow_profile(train: train, validation: validation)
        if !extends.nil?
          base_profile = { train: {}, validation: {} }
          base_names = extends.is_a?(Array) ? extends : [extends]
          base_names.each do |base_name|
            base_key = base_name.to_sym
            unless @dataflow_profiles.key?(base_key)
              raise ArgumentError, "unknown dataflow profile: #{base_name.inspect}"
            end

            base_profile = __dsl_compose_dataflow_profile(base_profile, @dataflow_profiles.fetch(base_key))
          end
          profile = __dsl_compose_dataflow_profile(base_profile, profile)
        end

        @dataflow_profiles[name.to_sym] = profile
        self
      end

      def use_dataflow(name, **overrides)
        profile = __dsl_resolve_dataflow_profile(name)
        profile_overrides, direct_overrides = __dsl_normalize_dataflow_overrides(overrides)
        resolved = __dsl_compose_dataflow_profile(profile, profile_overrides)
        __dsl_fit_kwargs_from_dataflow(resolved).merge(__dsl_normalize_fit_option_keys(direct_overrides))
      end

      def register_hook_pack(name, callable = nil, &block)
        if !callable.nil? && block_given?
          raise ArgumentError, "register_hook_pack accepts either a callable argument or block, not both"
        end

        pack = callable.nil? ? block : callable
        unless pack.respond_to?(:call)
          raise ArgumentError, "register_hook_pack requires a callable argument or block"
        end

        @hook_packs[name.to_sym] = pack
        self
      end

      def use_hook_pack(name, **options)
        key = name.to_sym
        unless @hook_packs.key?(key)
          raise ArgumentError, "unknown hook pack: #{name.inspect}"
        end

        __dsl_call_hook_pack(@hook_packs.fetch(key), options)
        self
      end

      def register_metric(name, callable = nil, &block)
        if !callable.nil? && block_given?
          raise ArgumentError, "register_metric accepts either a callable argument or block, not both"
        end

        metric_callable = callable.nil? ? block : callable
        unless metric_callable.respond_to?(:call)
          raise ArgumentError, "register_metric requires a callable argument or block"
        end

        @metric_registry[name.to_sym] = metric_callable
        self
      end

      def register_task(name, **defaults)
        @task_presets[name.to_sym] = __dsl_normalize_fit_option_keys(defaults)
        self
      end

      def fit_task(task, dataset, **overrides)
        fit(dataset, **__dsl_task_fit_options(task, overrides))
      end

      def fit_task_report(task, dataset, **overrides)
        fit_report(dataset, **__dsl_task_fit_options(task, overrides))
      end

      def artifact_policy(checkpoint: nil, retention: nil, resume: UNSET, run_bundle: nil)
        if checkpoint.nil? && retention.nil? && resume.equal?(UNSET) && run_bundle.nil?
          return __dsl_clone_config_value(@artifact_policy_config)
        end

        unless checkpoint.nil?
          @artifact_policy_config[:checkpoint] = __dsl_normalize_artifact_checkpoint_policy(checkpoint)
        end
        unless retention.nil?
          @artifact_policy_config[:retention] = __dsl_normalize_artifact_retention_policy(retention)
        end
        @artifact_policy_config[:resume] = resume unless resume.equal?(UNSET)
        unless run_bundle.nil?
          @artifact_policy_config[:run_bundle] = __dsl_normalize_artifact_run_bundle_policy(run_bundle)
        end
        self
      end

      def checkpoint_history
        __dsl_clone_config_value(@checkpoint_history)
      end

      def batch_schema(spec = UNSET, train: UNSET, validation: UNSET, **schema_kwargs)
        unless schema_kwargs.empty?
          if spec.equal?(UNSET) && train.equal?(UNSET) && validation.equal?(UNSET)
            spec = schema_kwargs
          else
            raise ArgumentError, "batch_schema keyword mappings cannot be combined with positional or split-specific forms"
          end
        end

        if !spec.equal?(UNSET) && (!train.equal?(UNSET) || !validation.equal?(UNSET))
          raise ArgumentError, "batch_schema accepts either positional spec or split-specific train:/validation: overrides"
        end
        if spec.equal?(UNSET) && train.equal?(UNSET) && validation.equal?(UNSET)
          return __dsl_clone_config_value(@batch_schemas)
        end

        if !spec.equal?(UNSET)
          normalized = __dsl_normalize_batch_schema_spec(spec)
          @batch_schemas[:train] = normalized
          @batch_schemas[:validation] = normalized
          return self
        end

        unless train.equal?(UNSET)
          @batch_schemas[:train] = __dsl_normalize_batch_schema_spec(train)
        end
        unless validation.equal?(UNSET)
          @batch_schemas[:validation] = __dsl_normalize_batch_schema_spec(validation)
        end
        self
      end

      def register_collate(name, spec = nil, extends: nil, &block)
        if !spec.nil? && block_given?
          raise ArgumentError, "register_collate accepts either a spec argument or block, not both"
        end

        key = name.to_sym
        resolved_spec = block_given? ? block : spec
        if resolved_spec.nil?
          raise ArgumentError, "register_collate requires a collate spec or block"
        end

        if !extends.nil?
          base_keys = extends.is_a?(Array) ? extends : [extends]
          base_spec = nil
          base_keys.each do |base_name|
            base_key = base_name.to_sym
            unless @collate_registry.key?(base_key)
              raise ArgumentError, "unknown base collate schema: #{base_name.inspect}"
            end

            current_base = @collate_registry.fetch(base_key)
            base_spec = base_spec.nil? ? current_base : __dsl_compose_collate(base_spec, current_base)
          end
          resolved_spec = __dsl_compose_collate(base_spec, resolved_spec)
        end

        @collate_registry[key] = resolved_spec
        self
      end

      def run_bundle(report:, config: {}, schema_version: "mlx_dsl_run_bundle_v1")
        unless report.is_a?(Hash)
          raise ArgumentError, "run_bundle requires report to be a Hash from fit_report"
        end

        {
          "format" => schema_version.to_s,
          "generated_at" => Time.now.utc.iso8601,
          "trainer" => {
            "monitor_name" => report["monitor_name"],
            "epochs_target" => report["epochs_target"],
            "epochs_completed" => report["epochs_completed"],
            "resume_from" => report["resume_from"],
            "resumed_from_epoch" => report["resumed_from_epoch"]
          },
          "config" => config || {},
          "report" => report,
          "checkpoint" => __dsl_deep_copy(@last_checkpoint_snapshot)
        }
      end

      def save_run_bundle(path, report:, config: {}, schema_version: "mlx_dsl_run_bundle_v1")
        bundle = run_bundle(
          report: report,
          config: config,
          schema_version: schema_version
        )
        dir = File.dirname(path.to_s)
        FileUtils.mkdir_p(dir) unless dir.nil? || dir.empty? || dir == "."
        File.binwrite(path, JSON.pretty_generate(bundle))
        path
      end

      def resume_payload_from_bundle(bundle_or_path)
        bundle = __dsl_resolve_run_bundle(bundle_or_path)
        checkpoint = bundle["checkpoint"]
        unless checkpoint.is_a?(Hash)
          raise ArgumentError, "run bundle does not include checkpoint snapshot"
        end

        metadata = checkpoint["metadata"]
        unless metadata.is_a?(Hash)
          raise ArgumentError, "run bundle checkpoint metadata is missing or invalid"
        end

        { "metadata" => __dsl_deep_copy(metadata) }
      end

      private

      def __dsl_clone_trainer
        cloned = self.class.new(**@__dsl_init_options, &@loss_block)
        hooks = Hash.new { |h, k| h[k] = [] }
        @hooks.each do |event, entries|
          hooks[event] = entries.map(&:dup)
        end
        cloned.instance_variable_set(:@hooks, hooks)
        cloned.instance_variable_set(:@hook_order, @hook_order)
        cloned.instance_variable_set(:@collate_registry, __dsl_clone_config_value(@collate_registry))
        cloned.instance_variable_set(:@fit_defaults, __dsl_clone_config_value(@fit_defaults))
        cloned.instance_variable_set(:@fit_presets, __dsl_clone_config_value(@fit_presets))
        cloned.instance_variable_set(:@batch_schemas, __dsl_clone_config_value(@batch_schemas))
        cloned.instance_variable_set(:@dataflow_profiles, __dsl_clone_config_value(@dataflow_profiles))
        cloned.instance_variable_set(:@hook_packs, __dsl_clone_config_value(@hook_packs))
        cloned.instance_variable_set(:@metric_registry, __dsl_clone_config_value(@metric_registry))
        cloned.instance_variable_set(:@task_presets, __dsl_clone_config_value(@task_presets))
        cloned.instance_variable_set(:@artifact_policy_config, __dsl_clone_config_value(@artifact_policy_config))
        cloned.instance_variable_set(:@checkpoint_history, __dsl_clone_config_value(@checkpoint_history))
        cloned.instance_variable_set(:@last_checkpoint_snapshot, __dsl_clone_config_value(@last_checkpoint_snapshot))
        cloned
      end

      def __dsl_clone_config_value(value)
        case value
        when Hash
          value.each_with_object({}) do |(key, item), out|
            out[key] = __dsl_clone_config_value(item)
          end
        when Array
          value.map { |item| __dsl_clone_config_value(item) }
        else
          value
        end
      end

      def __dsl_expand_split_plan(dataset, raw_options)
        return [dataset, raw_options] unless defined?(MLX::DSL::SplitPlan) && dataset.is_a?(MLX::DSL::SplitPlan)

        train_dataset, plan_options = dataset.to_fit_inputs
        merged = raw_options.dup
        plan_options.each do |key, value|
          next unless merged.key?(key)
          next unless merged.fetch(key).equal?(UNSET)

          merged[key] = value
        end
        [train_dataset, merged]
      end

      def __dsl_normalize_artifact_checkpoint_policy(checkpoint)
        unless checkpoint.is_a?(Hash)
          raise ArgumentError, "artifact checkpoint policy must be a Hash"
        end

        normalized = checkpoint.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        unknown = normalized.keys - %i[path strategy every]
        unless unknown.empty?
          raise ArgumentError, "artifact checkpoint policy has unsupported key(s): #{unknown.map(&:inspect).join(', ')}"
        end
        if normalized.key?(:strategy)
          strategy = normalized.fetch(:strategy).to_sym
          unless %i[latest best every].include?(strategy)
            raise ArgumentError, "artifact checkpoint strategy must be :latest, :best, or :every"
          end

          normalized[:strategy] = strategy
        end
        if normalized.key?(:every)
          every = normalized.fetch(:every).to_i
          raise ArgumentError, "artifact checkpoint every must be positive" if every <= 0

          normalized[:every] = every
        end
        normalized
      end

      def __dsl_normalize_artifact_retention_policy(retention)
        unless retention.is_a?(Hash)
          raise ArgumentError, "artifact retention policy must be a Hash"
        end

        normalized = retention.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        unknown = normalized.keys - %i[keep_last_n]
        unless unknown.empty?
          raise ArgumentError, "artifact retention policy has unsupported key(s): #{unknown.map(&:inspect).join(', ')}"
        end
        if normalized.key?(:keep_last_n)
          keep = normalized.fetch(:keep_last_n).to_i
          raise ArgumentError, "artifact retention keep_last_n must be non-negative" if keep.negative?

          normalized[:keep_last_n] = keep
        end
        normalized
      end

      def __dsl_normalize_artifact_run_bundle_policy(run_bundle)
        unless run_bundle.is_a?(Hash)
          raise ArgumentError, "artifact run_bundle policy must be a Hash"
        end

        normalized = run_bundle.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        unknown = normalized.keys - %i[enabled path config]
        unless unknown.empty?
          raise ArgumentError, "artifact run_bundle policy has unsupported key(s): #{unknown.map(&:inspect).join(', ')}"
        end
        normalized[:enabled] = !!normalized.fetch(:enabled, false)
        normalized[:config] = {} if normalized[:config].nil?
        normalized
      end

      def __dsl_resolve_artifact_policy(checkpoint_path:, save_best:, resume_from:, monitor_mode:)
        checkpoint_policy = @artifact_policy_config.fetch(:checkpoint, {})
        retention_policy = @artifact_policy_config.fetch(:retention, {})
        run_bundle_policy = @artifact_policy_config.fetch(:run_bundle, {})
        resume_policy = @artifact_policy_config.fetch(:resume, nil)

        resolved_checkpoint_path = checkpoint_path
        if (resolved_checkpoint_path.nil? || resolved_checkpoint_path.to_s.empty?) && checkpoint_policy.key?(:path)
          resolved_checkpoint_path = checkpoint_policy.fetch(:path)
        end

        resolved_save_best = save_best
        strategy = checkpoint_policy[:strategy]
        case strategy
        when :best
          resolved_save_best = true
        when :latest, :every
          resolved_save_best = false
        end

        resolved_resume = resume_from
        if (resolved_resume.nil? || resolved_resume.to_s.empty?) && !resume_policy.nil?
          resolved_resume = __dsl_policy_resume_source(resume_policy, monitor_mode: monitor_mode)
        end

        {
          checkpoint_path: resolved_checkpoint_path,
          save_best: resolved_save_best,
          checkpoint_every: checkpoint_policy[:every],
          keep_last_n: retention_policy[:keep_last_n],
          resume_from: resolved_resume,
          run_bundle: __dsl_clone_config_value(run_bundle_policy),
          payload: __dsl_stringify_keys(__dsl_clone_config_value(@artifact_policy_config))
        }
      end

      def __dsl_policy_resume_source(policy_resume, monitor_mode:)
        case policy_resume
        when :latest
          entry = @checkpoint_history.last
          return nil if entry.nil?

          { "metadata" => __dsl_deep_copy(entry.fetch("metadata")) }
        when :best
          entry = __dsl_best_checkpoint_from_history(monitor_mode)
          return nil if entry.nil?

          { "metadata" => __dsl_deep_copy(entry.fetch("metadata")) }
        else
          policy_resume
        end
      end

      def __dsl_best_checkpoint_from_history(monitor_mode)
        rows = @checkpoint_history.select do |row|
          row.is_a?(Hash) && row.key?("metadata") && row.fetch("metadata").is_a?(Hash)
        end
        return nil if rows.empty?

        comparator = monitor_mode.to_sym
        case comparator
        when :max
          rows.max_by { |row| row.fetch("monitor_value").to_f }
        else
          rows.min_by { |row| row.fetch("monitor_value").to_f }
        end
      end

      def __dsl_auto_save_run_bundle(run_bundle_policy, report_payload)
        return nil unless run_bundle_policy.is_a?(Hash)
        return nil unless run_bundle_policy.fetch(:enabled, false)

        path = run_bundle_policy[:path]
        return nil if path.nil? || path.to_s.empty?

        save_run_bundle(
          path,
          report: report_payload,
          config: run_bundle_policy.fetch(:config, {})
        )
      end

      def __dsl_stringify_keys(value)
        case value
        when Hash
          value.each_with_object({}) do |(key, item), out|
            out[key.to_s] = __dsl_stringify_keys(item)
          end
        when Array
          value.map { |item| __dsl_stringify_keys(item) }
        else
          value
        end
      end

      def __dsl_normalize_fit_option_keys(options)
        normalized = (options || {}).each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        unknown = normalized.keys - FIT_OPTION_DEFAULTS.keys
        unless unknown.empty?
          raise ArgumentError, "unsupported fit option(s): #{unknown.map(&:inspect).join(', ')}"
        end

        normalized
      end

      def __dsl_resolve_fit_options(raw_options)
        normalized_raw = __dsl_normalize_fit_option_keys(raw_options)
        normalized_defaults = __dsl_normalize_fit_option_keys(@fit_defaults)
        FIT_OPTION_DEFAULTS.each_with_object({}) do |(key, fallback), out|
          if normalized_raw.fetch(key).equal?(UNSET)
            out[key] = if normalized_defaults.key?(key)
              normalized_defaults.fetch(key)
            else
              __dsl_clone_config_value(fallback)
            end
          else
            out[key] = normalized_raw.fetch(key)
          end
        end
      end

      def __dsl_merge_fit_preset(name, overrides)
        key = name.to_sym
        unless @fit_presets.key?(key)
          raise ArgumentError, "unknown fit preset: #{name.inspect}"
        end

        __dsl_normalize_fit_option_keys(@fit_defaults)
          .merge(__dsl_normalize_fit_option_keys(@fit_presets.fetch(key)))
          .merge(__dsl_normalize_fit_option_keys(overrides))
      end

      def __dsl_builtin_task_presets
        {
          classification: {
            collate: :xy,
            monitor: :epoch_loss,
            monitor_mode: :min
          },
          regression: {
            collate: :xy,
            monitor: :epoch_loss,
            monitor_mode: :min
          },
          language_modeling: {
            collate: :xy,
            monitor: :perplexity,
            monitor_mode: :min,
            metric: ->(context) { Math.exp(context.fetch(:epoch_loss).to_f) }
          }
        }
      end

      def __dsl_task_fit_options(task, overrides)
        key = task.to_sym
        unless @task_presets.key?(key)
          raise ArgumentError, "unknown task preset: #{task.inspect}"
        end

        __dsl_normalize_fit_option_keys(@task_presets.fetch(key))
          .merge(__dsl_normalize_fit_option_keys(overrides))
      end

      def __dsl_resolve_dataflow_profile(name)
        key = name.to_sym
        unless @dataflow_profiles.key?(key)
          raise ArgumentError, "unknown dataflow profile: #{name.inspect}"
        end

        __dsl_clone_config_value(@dataflow_profiles.fetch(key))
      end

      def __dsl_normalize_dataflow_profile(train:, validation:)
        {
          train: __dsl_normalize_dataflow_split(train, split: :train),
          validation: __dsl_normalize_dataflow_split(validation, split: :validation)
        }
      end

      def __dsl_normalize_dataflow_split(spec, split:)
        return {} if spec.nil?
        unless spec.is_a?(Hash)
          raise ArgumentError, "#{split} dataflow spec must be a Hash or nil"
        end

        normalized = spec.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        allowed = %i[collate transform limit reduce]
        unknown = normalized.keys - allowed
        unless unknown.empty?
          raise ArgumentError, "#{split} dataflow spec has unsupported key(s): #{unknown.map(&:inspect).join(', ')}"
        end

        normalized
      end

      def __dsl_compose_dataflow_profile(base, overlay)
        {
          train: base.fetch(:train, {}).merge(overlay.fetch(:train, {})),
          validation: base.fetch(:validation, {}).merge(overlay.fetch(:validation, {}))
        }
      end

      def __dsl_normalize_dataflow_overrides(overrides)
        profile_overrides = { train: {}, validation: {} }
        direct_overrides = {}
        (overrides || {}).each do |key, value|
          key = key.to_sym
          case key
          when :train, :validation
            profile_overrides[key] = __dsl_normalize_dataflow_split(value, split: key)
          else
            direct_overrides[key] = value
          end
        end
        [profile_overrides, direct_overrides]
      end

      def __dsl_fit_kwargs_from_dataflow(profile)
        {
          collate: profile.fetch(:train).fetch(:collate, UNSET),
          train_transform: profile.fetch(:train).fetch(:transform, UNSET),
          limit: profile.fetch(:train).fetch(:limit, UNSET),
          reduce: profile.fetch(:train).fetch(:reduce, UNSET),
          validation_collate: profile.fetch(:validation).fetch(:collate, UNSET),
          validation_transform: profile.fetch(:validation).fetch(:transform, UNSET),
          validation_limit: profile.fetch(:validation).fetch(:limit, UNSET),
          validation_reduce: profile.fetch(:validation).fetch(:reduce, UNSET)
        }.each_with_object({}) do |(key, value), out|
          out[key] = value unless value.equal?(UNSET)
        end
      end

      def __dsl_call_hook_pack(pack, options)
        values = { trainer: self, options: options || {} }
        (options || {}).each do |key, value|
          values[key.to_sym] = value
        end
        if !pack.respond_to?(:parameters) || pack.parameters.empty?
          return instance_exec(&pack) if pack.is_a?(Proc)

          return pack.call
        end

        params = pack.parameters
        args = __dsl_build_positional_args(
          params,
          values,
          [[:trainer, self], [:options, options || {}]],
          "hook pack"
        )
        kwargs = __dsl_build_keyword_args(params, values, "hook pack")
        return pack.call(*args) if kwargs.empty?

        pack.call(*args, **kwargs)
      end

      def __dsl_normalize_sync(sync)
        mode = sync.nil? ? :none : sync.to_sym
        return mode if %i[none step epoch].include?(mode)

        raise ArgumentError, "trainer sync must be one of :none, :step, or :epoch"
      end

      def __dsl_build_train_step(model, optimizer:, clip_grad_norm:, compile:, &loss_block)
        params = model.method(:train_step).parameters
        accepts_keyrest = params.any? { |type, _name| type == :keyrest }
        accepts_keyword = lambda do |key|
          accepts_keyrest || params.any? do |type, name|
            (type == :key || type == :keyreq) && name == key
          end
        end

        kwargs = {
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm
        }
        kwargs[:compile] = compile if accepts_keyword.call(:compile)
        kwargs[:sync] = (@sync_mode == :step ? :step : :none) if accepts_keyword.call(:sync)

        model.train_step(**kwargs, &loss_block)
      end

      def __dsl_resume_state(resume_from, monitor_name)
        return __dsl_empty_resume_state(monitor_name) if resume_from.nil? || resume_from.to_s.empty?

        source = resume_from
        source = __dsl_call_resume_loader(source, monitor_name) if source.respond_to?(:call)
        return __dsl_empty_resume_state(monitor_name) if source.nil?

        if source.is_a?(Hash)
          payload = if __dsl_run_bundle_hash?(source)
            resume_payload_from_bundle(source)
          else
            source
          end
          resume_path = nil
        else
          source_path = source.to_s
          bundle_payload = __dsl_resume_payload_from_run_bundle_path(source_path)
          if bundle_payload.nil?
            unless @model.respond_to?(:load_checkpoint)
              raise ArgumentError, "resume_from requires model to implement #load_checkpoint"
            end

            payload = __dsl_load_checkpoint_for_resume(source)
            resume_path = source_path
          else
            payload = bundle_payload
            resume_path = source_path
          end
        end
        unless payload.is_a?(Hash)
          raise ArgumentError, "resume checkpoint payload must be a Hash"
        end

        metadata = payload["metadata"]
        metadata = {} unless metadata.is_a?(Hash)

        checkpoint_epoch_value = __dsl_resume_state_value(payload, metadata, "epoch")
        checkpoint_epoch = checkpoint_epoch_value.nil? ? nil : checkpoint_epoch_value.to_i
        start_epoch = checkpoint_epoch.nil? ? 0 : checkpoint_epoch + 1
        best_metric = __dsl_resume_state_value(payload, metadata, "best_metric")
        stale_epochs_value = __dsl_resume_state_value(payload, metadata, "stale_epochs")
        stale_epochs = stale_epochs_value.nil? ? 0 : stale_epochs_value.to_i
        if stale_epochs.negative?
          raise ArgumentError, "resume checkpoint stale_epochs must be non-negative"
        end

        resume_monitor_name = __dsl_resume_state_value(payload, metadata, "monitor_name")
        if !resume_monitor_name.nil? && resume_monitor_name.to_s != monitor_name.to_s
          raise ArgumentError,
                "resume checkpoint monitor_name #{resume_monitor_name.inspect} does not match requested monitor #{monitor_name.inspect}"
        end

        {
          path: resume_path,
          checkpoint_epoch: checkpoint_epoch,
          start_epoch: start_epoch,
          best_metric: best_metric,
          stale_epochs: stale_epochs,
          monitor_name: monitor_name.to_s
        }
      end

      def __dsl_run_bundle_hash?(value)
        value.is_a?(Hash) && value.key?("checkpoint") && value.key?("report")
      end

      def __dsl_resume_payload_from_run_bundle_path(path)
        return nil if path.nil? || path.empty?
        return nil unless File.file?(path)

        bundle = JSON.parse(File.binread(path))
        return nil unless __dsl_run_bundle_hash?(bundle)

        resume_payload_from_bundle(bundle)
      rescue JSON::ParserError
        nil
      end

      def __dsl_empty_resume_state(monitor_name)
        {
          path: nil,
          checkpoint_epoch: nil,
          start_epoch: 0,
          best_metric: nil,
          stale_epochs: 0,
          monitor_name: monitor_name.to_s
        }
      end

      def __dsl_call_resume_loader(loader, monitor_name)
        values = {
          trainer: self,
          model: @model,
          optimizer: @optimizer,
          monitor_name: monitor_name.to_s
        }
        return loader.call unless loader.respond_to?(:parameters)

        params = loader.parameters
        return loader.call if params.empty?

        args = __dsl_build_positional_args(
          params,
          values,
          [[:trainer, self], [:model, @model], [:optimizer, @optimizer], [:monitor_name, monitor_name.to_s]],
          "resume loader"
        )
        kwargs = __dsl_build_keyword_args(params, values, "resume loader")
        return loader.call(*args) if kwargs.empty?

        loader.call(*args, **kwargs)
      end

      def __dsl_load_checkpoint_for_resume(path)
        params = @model.method(:load_checkpoint).parameters
        accepts_keyrest = params.any? { |type, _name| type == :keyrest }
        accepts_keyword = lambda do |key|
          accepts_keyrest || params.any? do |type, name|
            (type == :key || type == :keyreq) && name == key
          end
        end

        kwargs = {}
        kwargs[:optimizer] = @optimizer if accepts_keyword.call(:optimizer)
        kwargs[:strict] = true if accepts_keyword.call(:strict)
        kwargs[:format] = nil if accepts_keyword.call(:format)
        return @model.load_checkpoint(path) if kwargs.empty?

        @model.load_checkpoint(path, **kwargs)
      end

      def __dsl_resume_state_value(payload, metadata, key)
        return metadata[key] if metadata.key?(key)

        payload[key]
      end

      def emit(event, context)
        @hooks[event.to_sym].sort_by { |entry| [entry.fetch(:priority), entry.fetch(:order)] }.each do |entry|
          entry[:invocations] += 1
          if !entry[:every].nil? && ((entry[:invocations] - 1) % entry[:every]).nonzero?
            next
          end
          if entry[:once] && entry[:fired]
            next
          end
          unless __dsl_hook_condition_met?(entry[:if], context)
            next
          end

          entry.fetch(:hook).call(context)
          entry[:fired] = true if entry[:once]
        end
      end

      def __dsl_hook_condition_met?(condition, context)
        return true if condition.nil?
        return !!condition unless condition.respond_to?(:call)
        return !!condition.call unless condition.respond_to?(:parameters)

        params = condition.parameters
        return !!condition.call if params.empty?

        if params.any? { |type, _name| type == :keyrest || type == :key || type == :keyreq }
          return !!condition.call(context: context)
        end

        !!condition.call(context)
      end

      def __dsl_sync_epoch(loss)
        return unless defined?(MLX::Core) && MLX::Core.respond_to?(:eval)

        targets = []
        targets << loss unless loss.nil?
        targets << @model.parameters if @model.respond_to?(:parameters)
        targets << @optimizer.state if @optimizer.respond_to?(:state)
        return if targets.empty?

        MLX::Core.eval(*targets)
      end

      def __dsl_apply_collate(collate, batch, kind:, epoch:, batch_index:)
        collate = __dsl_resolve_registered_collate(collate, kind: kind)
        collate = __dsl_auto_collate_spec(kind, batch) if collate.to_s == "auto"
        return batch if collate.nil?
        if collate.respond_to?(:call)
          return __dsl_call_collate_callable(
            collate,
            batch,
            kind: kind,
            epoch: epoch,
            batch_index: batch_index
          )
        end

        case collate
        when String, Symbol
          __dsl_apply_named_collate(collate.to_sym, batch, kind: kind)
        when Hash
          __dsl_apply_mapping_collate(
            collate,
            batch,
            kind: kind,
            epoch: epoch,
            batch_index: batch_index
          )
        else
          raise ArgumentError, "#{kind} collate must be a Proc, Symbol/String, Hash, or nil"
        end
      end

      def __dsl_effective_collate(collate, bind, batch, kind:)
        return collate unless collate.nil?
        return nil if bind.nil?

        __dsl_bind_to_collate(bind, batch, kind: kind)
      end

      def __dsl_bind_to_collate(bind, batch, kind:)
        case bind
        when true
          __dsl_infer_bind_mapping(kind, batch)
        when String, Symbol
          return __dsl_infer_bind_mapping(kind, batch) if bind.to_s == "auto"

          bind
        when Hash
          bind
        when Proc
          __dsl_call_collate_callable(
            bind,
            batch,
            kind: kind,
            epoch: 0,
            batch_index: 0,
            label: "#{kind} bind"
          )
        else
          raise ArgumentError, "#{kind} bind must be :auto/true, collate spec, or hash mapping"
        end
      end

      def __dsl_infer_bind_mapping(kind, batch)
        target_params = __dsl_keyword_names_for_callable(kind == :train ? @step.method(:call) : @loss_block)
        return :xy if target_params.empty? && batch.is_a?(Array) && batch.length >= 2
        return :x if target_params.empty? && !batch.is_a?(Hash)

        target_params.each_with_index.each_with_object({}) do |(name, index), out|
          out[name] = __dsl_infer_bind_selector(batch, name, index)
        end
      end

      def __dsl_keyword_names_for_callable(callable)
        return [] unless callable.respond_to?(:parameters)

        callable.parameters.each_with_object([]) do |(type, name), out|
          out << name if (type == :key || type == :keyreq) && !name.nil?
        end
      end

      def __dsl_infer_bind_selector(batch, name, index)
        if batch.is_a?(Hash)
          candidates = __dsl_bind_candidates_for(name)
          found = candidates.find do |candidate|
            batch.key?(candidate) || batch.key?(candidate.to_s) || batch.key?(candidate.to_sym)
          end
          return found unless found.nil?

          return name
        end

        return index if batch.respond_to?(:[]) && !batch.is_a?(Hash)

        name
      end

      def __dsl_bind_candidates_for(name)
        base = [name, name.to_s, name.to_sym]
        aliases = case name.to_sym
        when :x
          %i[input inputs feature features token tokens]
        when :y
          %i[target targets label labels output outputs]
        else
          []
        end
        base + aliases + aliases.map(&:to_s)
      end

      def __dsl_auto_collate_spec(kind, batch)
        schema = @batch_schemas.fetch(kind, nil)
        return schema unless schema.nil?

        if batch.is_a?(Hash)
          has_x = batch.key?(:x) || batch.key?("x")
          has_y = batch.key?(:y) || batch.key?("y")
          return :xy if has_x && has_y
          return { x: :x } if has_x

          return nil
        end

        return :xy if batch.is_a?(Array) && batch.length >= 2

        :x
      end

      def __dsl_normalize_batch_schema_spec(spec)
        return nil if spec.nil?
        return spec if spec.respond_to?(:call)

        case spec
        when String, Symbol, Hash
          spec
        else
          raise ArgumentError, "batch schema must be a collate spec (Symbol/String, Hash, Proc, or nil)"
        end
      end

      def __dsl_resolve_registered_collate(collate, kind:)
        current = collate
        seen = []
        while current.is_a?(String) || current.is_a?(Symbol)
          key = current.to_sym
          break unless @collate_registry.key?(key)
          if seen.include?(key)
            cycle = (seen + [key]).map(&:inspect).join(" -> ")
            raise ArgumentError, "cyclic #{kind} collate registry reference: #{cycle}"
          end

          seen << key
          current = @collate_registry.fetch(key)
        end
        current
      end

      def __dsl_compose_collate(base_spec, overlay_spec)
        if base_spec.is_a?(Hash) && overlay_spec.is_a?(Hash)
          return base_spec.merge(overlay_spec)
        end

        raise ArgumentError, "collate extends composition supports Hash schemas only"
      end

      def __dsl_apply_named_collate(name, batch, kind:)
        case name
        when :xy
          return batch if batch.is_a?(Hash) && (batch.key?(:x) || batch.key?("x")) && (batch.key?(:y) || batch.key?("y"))
          unless batch.is_a?(Array) && batch.length >= 2
            raise ArgumentError, "#{kind} collate :xy expects a 2-item array batch"
          end

          { x: batch[0], y: batch[1] }
        when :x
          { x: batch }
        else
          raise ArgumentError, "unknown #{kind} collate schema: #{name.inspect}"
        end
      end

      def __dsl_apply_mapping_collate(mapping, batch, kind:, epoch:, batch_index:)
        mapping.each_with_object({}) do |(out_key, selector), out|
          out[out_key] = __dsl_collate_select(
            batch,
            selector,
            kind: kind,
            epoch: epoch,
            batch_index: batch_index
          )
        end
      end

      def __dsl_collate_select(batch, selector, kind:, epoch:, batch_index:)
        case selector
        when Integer
          unless batch.respond_to?(:[]) && !batch.is_a?(Hash)
            raise ArgumentError, "#{kind} collate integer selector requires indexable non-hash batch"
          end
          batch[selector]
        when String, Symbol
          __dsl_collate_fetch_key(batch, selector, kind: kind)
        when Proc
          __dsl_call_collate_callable(
            selector,
            batch,
            kind: kind,
            epoch: epoch,
            batch_index: batch_index,
            label: "#{kind} collate selector"
          )
        when Array
          __dsl_collate_select_path(batch, selector, kind: kind)
        else
          raise ArgumentError, "#{kind} collate selector must be Integer, String/Symbol, Proc, or Array path"
        end
      end

      def __dsl_call_collate_callable(callable, batch, kind:, epoch:, batch_index:, label: nil)
        values = {
          batch: batch,
          epoch: epoch,
          batch_index: batch_index,
          kind: kind,
          trainer: self
        }
        return callable.call(batch) unless callable.respond_to?(:parameters)

        params = callable.parameters
        return callable.call(batch) if params.empty?

        callable_label = label || "#{kind} collate"
        args = __dsl_build_positional_args(
          params,
          values,
          [[:batch, batch], [:epoch, epoch], [:batch_index, batch_index], [:kind, kind], [:trainer, self]],
          callable_label
        )
        kwargs = __dsl_build_keyword_args(params, values, callable_label)
        return callable.call(*args) if kwargs.empty?

        callable.call(*args, **kwargs)
      end

      def __dsl_collate_select_path(batch, selector_path, kind:)
        if selector_path.empty?
          raise ArgumentError, "#{kind} collate selector path must not be empty"
        end

        selector_path.each_with_index.reduce(batch) do |current, (selector, depth)|
          __dsl_collate_select_path_segment(current, selector, selector_path, depth, kind: kind)
        end
      end

      def __dsl_collate_select_path_segment(current, selector, selector_path, depth, kind:)
        case selector
        when Integer
          unless current.respond_to?(:[]) && !current.is_a?(Hash)
            raise ArgumentError, "#{kind} collate path #{selector_path.inspect} expected indexable non-hash at depth #{depth}"
          end
          current[selector]
        when String, Symbol
          __dsl_collate_fetch_key(
            current,
            selector,
            kind: kind,
            context: "in path #{selector_path.inspect} at depth #{depth}"
          )
        else
          raise ArgumentError, "#{kind} collate path #{selector_path.inspect} contains unsupported selector #{selector.inspect}"
        end
      end

      def __dsl_collate_fetch_key(batch, selector, kind:, context: nil)
        unless batch.is_a?(Hash)
          extra = context.nil? ? "" : " #{context}"
          raise ArgumentError, "#{kind} collate key selector #{selector.inspect} requires hash batch#{extra}"
        end

        return batch.fetch(selector) if batch.key?(selector)

        str_key = selector.to_s
        return batch.fetch(str_key) if batch.key?(str_key)

        sym_key = str_key.to_sym
        return batch.fetch(sym_key) if batch.key?(sym_key)

        extra = context.nil? ? "" : " #{context}"
        raise ArgumentError, "#{kind} collate key selector #{selector.inspect} was not found in batch#{extra}"
      end

      def __dsl_run_batch(batch, epoch:, batch_index:, kind:)
        if batch.is_a?(Hash)
          @step.call(**__dsl_normalize_batch_kwargs(batch, label: "#{kind} batch"))
        elsif batch.is_a?(Array)
          @step.call(*batch)
        else
          @step.call(batch)
        end
      rescue StandardError => e
        __dsl_raise_batch_error!(e, kind: kind, epoch: epoch, batch_index: batch_index)
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
        metric_callable = __dsl_resolve_metric_callable(metric)
        return fallback if metric_callable.nil?
        return metric_callable.call(context) unless metric_callable.respond_to?(:parameters)

        params = metric_callable.parameters
        return metric_callable.call(context) if params.empty?

        values = {
          context: context,
          trainer: self
        }
        args = __dsl_build_positional_args(
          params,
          values,
          [[:context, context], [:trainer, self]],
          "metric callable"
        )
        kwargs = __dsl_build_keyword_args(params, values, "metric callable")
        return metric_callable.call(*args) if kwargs.empty?

        metric_callable.call(*args, **kwargs)
      end

      def __dsl_resolve_metric_callable(metric)
        return nil if metric.nil?
        return metric if metric.respond_to?(:call)
        return nil unless metric.is_a?(String) || metric.is_a?(Symbol)

        @metric_registry[metric.to_sym]
      end

      def __dsl_run_validation_batch(batch, epoch:, batch_index:, kind:)
        if batch.is_a?(Hash)
          @loss_block.call(**__dsl_normalize_batch_kwargs(batch, label: "#{kind} batch"))
        elsif batch.is_a?(Array)
          @loss_block.call(*batch)
        else
          @loss_block.call(batch)
        end
      rescue StandardError => e
        __dsl_raise_batch_error!(e, kind: kind, epoch: epoch, batch_index: batch_index)
      end

      def __dsl_raise_batch_error!(error, kind:, epoch:, batch_index:)
        prefix = "#{kind} batch failed at epoch #{epoch}, batch #{batch_index}"
        raise error.class, "#{prefix}: #{error.message}"
      end

      def __dsl_normalize_batch_kwargs(batch, label:)
        batch.each_with_object({}) do |(key, value), out|
          kw_key = __dsl_normalize_keyword_key(key, label: label)
          if out.key?(kw_key)
            raise ArgumentError, "#{label} contains duplicate keyword after normalization: #{kw_key.inspect}"
          end

          out[kw_key] = value
        end
      end

      def __dsl_normalize_keyword_key(key, label:)
        return key if key.is_a?(Symbol)
        return key.to_sym if key.respond_to?(:to_sym)

        raise ArgumentError, "#{label} key #{key.inspect} cannot be converted to keyword symbol"
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
        stale_epochs:,
        best_metric:,
        metadata:,
        checkpoint_every: nil,
        keep_last_n: nil
      )
        return false if path.nil? || path.to_s.empty?
        unless checkpoint_every.nil?
          every = checkpoint_every.to_i
          raise ArgumentError, "checkpoint every must be positive" if every <= 0
          return false if ((epoch + 1) % every).nonzero?
        end
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
        merged_metadata["stale_epochs"] = stale_epochs
        merged_metadata["best_metric"] = best_metric
        merged_metadata["next_epoch"] = epoch + 1

        @model.save_checkpoint(resolved_path, optimizer: @optimizer, metadata: merged_metadata)
        @last_checkpoint_snapshot = {
          "path" => resolved_path,
          "epoch" => epoch,
          "monitor_name" => monitor_name,
          "monitor_value" => monitor_value,
          "epoch_loss" => epoch_metric,
          "metadata" => __dsl_deep_copy(merged_metadata)
        }
        @checkpoint_history << __dsl_deep_copy(@last_checkpoint_snapshot)
        unless keep_last_n.nil?
          keep = keep_last_n.to_i
          raise ArgumentError, "artifact retention keep_last_n must be non-negative" if keep.negative?
          @checkpoint_history.shift while @checkpoint_history.length > keep
        end
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
        if path.respond_to?(:call)
          values = {
            epoch: epoch,
            next_epoch: epoch + 1,
            monitor: monitor_value,
            monitor_name: monitor_name,
            epoch_loss: epoch_metric,
            improved: improved,
            trainer: self,
            model: @model,
            optimizer: @optimizer
          }
          path = if !path.respond_to?(:parameters) || path.parameters.empty?
            path.call
          else
            args = __dsl_build_positional_args(
              path.parameters,
              values,
              [
                [:epoch, epoch],
                [:next_epoch, epoch + 1],
                [:monitor, monitor_value],
                [:monitor_name, monitor_name],
                [:epoch_loss, epoch_metric],
                [:improved, improved],
                [:trainer, self],
                [:model, @model],
                [:optimizer, @optimizer]
              ],
              "checkpoint path"
            )
            kwargs = __dsl_build_keyword_args(path.parameters, values, "checkpoint path")
            kwargs.empty? ? path.call(*args) : path.call(*args, **kwargs)
          end
          unless path.respond_to?(:to_str)
            raise ArgumentError, "checkpoint path callable must return a String-compatible path"
          end

          path = path.to_str
        end

        template = path.to_s
        return template unless template.include?("%{")

        template % {
          epoch: epoch,
          next_epoch: epoch + 1,
          monitor: monitor_value,
          monitor_name: monitor_name,
          epoch_loss: epoch_metric,
          improved: improved
        }
      rescue KeyError => e
        raise ArgumentError, "unsupported checkpoint template key: #{e.message}"
      end

      def __dsl_resolve_run_bundle(bundle_or_path)
        if bundle_or_path.is_a?(Hash)
          return bundle_or_path
        end

        path = bundle_or_path.to_s
        if path.empty?
          raise ArgumentError, "run bundle source must be a bundle hash or path"
        end

        JSON.parse(File.binread(path))
      end

      def __dsl_deep_copy(value)
        return nil if value.nil?

        Marshal.load(Marshal.dump(value))
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
          begin
            source.rewind
          rescue StandardError => e
            raise ArgumentError, "#{kind} dataset could not rewind for epoch #{epoch}: #{e.message}"
          end
        end

        source
      end

      def __dsl_call_dataset_factory(factory, epoch:, kind:)
        return factory.call unless factory.respond_to?(:parameters)

        params = factory.parameters
        values = {
          epoch: epoch,
          trainer: self,
          kind: kind
        }
        return factory.call if params.empty?

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

      def __dsl_resolve_loop_limit(limit, epoch:, kind:)
        return nil if limit.nil?

        raw = if limit.respond_to?(:call)
          values = {
            epoch: epoch,
            kind: kind,
            trainer: self
          }
          return limit.call if !limit.respond_to?(:parameters) || limit.parameters.empty?

          args = __dsl_build_positional_args(
            limit.parameters,
            values,
            [[:epoch, epoch], [:kind, kind], [:trainer, self]],
            "#{kind} limit"
          )
          kwargs = __dsl_build_keyword_args(limit.parameters, values, "#{kind} limit")
          kwargs.empty? ? limit.call(*args) : limit.call(*args, **kwargs)
        else
          limit
        end
        return nil if raw.nil?
        unless raw.respond_to?(:to_int)
          raise ArgumentError, "#{kind} limit must be an Integer, nil, or callable returning one"
        end

        value = raw.to_int
        raise ArgumentError, "#{kind} limit must be non-negative" if value.negative?

        value
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
