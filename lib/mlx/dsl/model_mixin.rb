# frozen_string_literal: true

require "set"
require "json"
require "fileutils"

module MLX
  module DSL
    module ModelMixin
      UNSET = Object.new.freeze

      class OptimizerGroupsBuilder
        def initialize
          @groups = []
        end

        def group(matcher = nil, &factory)
          raise ArgumentError, "group requires an optimizer block" unless block_given?

          optimizer = factory.call
          unless optimizer.is_a?(MLX::Optimizers::Optimizer)
            raise TypeError, "group block must return an MLX::Optimizers::Optimizer"
          end

          @groups << {
            optimizer: optimizer,
            filter: matcher_lambda(matcher)
          }
          optimizer
        end

        def build
          if @groups.empty?
            raise ArgumentError, "optimizer_groups requires at least one group"
          end

          return @groups.first[:optimizer] if @groups.length == 1

          MLX::Optimizers::MultiOptimizer.new(
            @groups.map { |entry| entry[:optimizer] },
            filters: @groups[0...-1].map { |entry| entry[:filter] }
          )
        end

        private

        def matcher_lambda(matcher)
          case matcher
          when nil
            lambda { |_path, _grad| true }
          when Regexp
            lambda { |path, _grad| matcher.match?(path.to_s) }
          when String, Symbol
            target = matcher.to_s
            lambda { |path, _grad| path.to_s == target }
          when Proc
            lambda do |path, grad|
              matcher.call(path.to_s, grad)
            end
          else
            raise ArgumentError, "unsupported group matcher: #{matcher.class}"
          end
        end
      end

      def self.included(base)
        base.extend(ClassMethods)
        base.prepend(Initializer)
      end

      module Initializer
        def initialize(*args, **kwargs, &block)
          dsl_options = __dsl_extract_declared_options(kwargs)
          unknown_kwargs = __dsl_unknown_kwargs_for_super(kwargs, method(__method__).super_method)
          unless unknown_kwargs.empty?
            names = unknown_kwargs.map(&:to_s).sort.join(", ")
            raise ArgumentError, "unknown option(s): #{names}"
          end

          super(*args, **kwargs, &block)
          __dsl_apply_option_values(dsl_options)
          __dsl_materialize_declarations!
        end
      end

      module ClassMethods
        def option(name, default: UNSET, required: UNSET)
          key = name.to_s
          required = default.equal?(UNSET) if required.equal?(UNSET)
          dsl_option_definitions[key] = { default: default, required: !!required }
        end

        def layer(name, factory = nil, *factory_args, **factory_kwargs, &block)
          if factory.nil? && !block_given?
            raise ArgumentError, "layer requires either a factory or block"
          end
          if !factory.nil? && block_given?
            raise ArgumentError, "layer cannot accept both a factory and block"
          end

          dsl_declarations << {
            kind: :layer,
            name: name.to_s,
            factory: factory,
            factory_args: factory_args,
            factory_kwargs: factory_kwargs,
            block: block
          }
        end

        def network(name, factory = nil, *factory_args, **factory_kwargs, &block)
          layer(name, factory, *factory_args, **factory_kwargs, &block)
        end

        def param(name, shape:, init: nil, dtype: UNSET)
          dsl_declarations << {
            kind: :param,
            name: name.to_s,
            shape: shape,
            init: init,
            dtype: dtype
          }
        end

        def buffer(name, shape:, init: nil, dtype: UNSET)
          dsl_declarations << {
            kind: :buffer,
            name: name.to_s,
            shape: shape,
            init: init,
            dtype: dtype
          }
        end

        def dsl_option_definitions
          @dsl_option_definitions ||= {}
        end

        def dsl_declarations
          @dsl_declarations ||= []
        end

        def inherited(subclass)
          super
          copied_options = dsl_option_definitions.each_with_object({}) do |(key, value), out|
            out[key] = value.dup
          end
          copied_declarations = dsl_declarations.map(&:dup)
          subclass.instance_variable_set(:@dsl_option_definitions, copied_options)
          subclass.instance_variable_set(:@dsl_declarations, copied_declarations)
        end
      end

      include TrainStepMethods

      def optimizer_groups(&block)
        raise ArgumentError, "optimizer_groups requires a block" unless block_given?

        builder = OptimizerGroupsBuilder.new
        builder.instance_eval(&block)
        builder.build
      end

      def trainer(optimizer:, clip_grad_norm: nil, compile: false, sync: :none, &loss_block)
        MLX::DSL::Trainer.new(
          model: self,
          optimizer: optimizer,
          clip_grad_norm: clip_grad_norm,
          compile: compile,
          sync: sync,
          &loss_block
        )
      end

      def save_checkpoint(path, optimizer: nil, metadata: {}, format: nil)
        checkpoint_format = __dsl_checkpoint_format(path, format)
        if checkpoint_format == :marshal
          __dsl_ensure_parent_dir!(path)
          payload = {
            "format" => "mlx_dsl_checkpoint_v1",
            "model" => __dsl_serialize_tree(parameters),
            "metadata" => metadata || {}
          }
          payload["optimizer"] = __dsl_serialize_tree(optimizer.state) unless optimizer.nil?

          File.binwrite(path, Marshal.dump(payload))
          return path
        end

        __dsl_save_native_checkpoint(path, checkpoint_format, optimizer: optimizer, metadata: metadata)
      end

      def load_checkpoint(path, optimizer: nil, strict: true, format: nil)
        resolved_path = __dsl_resolve_load_checkpoint_path(path, format)
        checkpoint_format = __dsl_checkpoint_format(resolved_path, format)
        if checkpoint_format == :marshal
          payload = Marshal.load(File.binread(resolved_path))
          unless payload.is_a?(Hash) && payload["format"] == "mlx_dsl_checkpoint_v1"
            raise ArgumentError, "unsupported checkpoint format"
          end

          model_state = __dsl_deserialize_tree(payload.fetch("model"))
          update(model_state, strict: strict)

          if !optimizer.nil? && payload.key?("optimizer")
            optimizer.state = __dsl_deserialize_tree(payload["optimizer"])
          end

          return payload
        end

        weights_path = __dsl_checkpoint_weights_path(resolved_path, checkpoint_format)
        load_weights(weights_path, strict: strict)
        payload = __dsl_load_native_checkpoint_payload(weights_path)

        if !optimizer.nil? && payload.key?("optimizer")
          optimizer.state = __dsl_deserialize_tree(payload["optimizer"])
        end

        payload
      end

      def train_mode
        previous = training
        train(true)
        return self unless block_given?

        yield(self)
      ensure
        train(previous) unless previous.nil?
      end

      def eval_mode
        previous = training
        eval
        return self unless block_given?

        yield(self)
      ensure
        train(previous) unless previous.nil?
      end

      def freeze_paths!(matcher, strict: false)
        selected = __dsl_select_paths(matcher)
        if strict && selected.empty?
          raise KeyError, "no parameter paths matched #{matcher.inspect}"
        end

        __dsl_toggle_paths(selected, freeze: true)
      end

      def unfreeze_paths!(matcher, strict: false)
        selected = __dsl_select_paths(matcher)
        if strict && selected.empty?
          raise KeyError, "no parameter paths matched #{matcher.inspect}"
        end

        __dsl_toggle_paths(selected, freeze: false)
      end

      def parameter_paths(matcher: nil)
        all_paths = MLX::Utils.tree_flatten(parameters, destination: {}).keys.sort
        return all_paths if matcher.nil?

        path_matcher = __dsl_path_matcher(matcher)
        all_paths.select { |path| path_matcher.call(path) }
      end

      def parameter_count
        __dsl_count_parameters(parameters)
      end

      def trainable_parameter_count
        __dsl_count_parameters(trainable_parameters)
      end

      def summary(as: :hash)
        total = parameter_count
        trainable = trainable_parameter_count
        payload = {
          "model_class" => self.class.name.to_s,
          "total_parameters" => total,
          "trainable_parameters" => trainable,
          "frozen_parameters" => total - trainable,
          "parameter_paths" => parameter_paths
        }
        case as.to_sym
        when :hash
          payload
        when :text
          __dsl_summary_text(payload)
        else
          raise ArgumentError, "summary :as must be :hash or :text"
        end
      end

      private

      def __dsl_summary_text(payload)
        [
          "model=#{payload.fetch('model_class')}",
          "total_parameters=#{payload.fetch('total_parameters')}",
          "trainable_parameters=#{payload.fetch('trainable_parameters')}",
          "frozen_parameters=#{payload.fetch('frozen_parameters')}",
          "parameter_paths=#{payload.fetch('parameter_paths').join(',')}"
        ].join("\n")
      end

      def __dsl_count_parameters(tree)
        flat = MLX::Utils.tree_flatten(tree, destination: {})
        flat.values.sum { |value| __dsl_leaf_numel(value) }
      end

      def __dsl_leaf_numel(value)
        if value.respond_to?(:shape)
          shape = value.shape
          if shape.is_a?(Array)
            return 1 if shape.empty?

            return shape.reduce(1) { |acc, dim| acc * dim.to_i }
          end
        end
        return value.size.to_i if value.respond_to?(:size)

        1
      end

      def __dsl_checkpoint_format(path, format)
        raw = if format.nil?
          ext = File.extname(path.to_s).delete_prefix(".").downcase
          ext.empty? ? "marshal" : ext
        else
          format.to_s.downcase
        end

        case raw
        when "marshal", "bin", "legacy", "dsl_v1"
          :marshal
        when "npz"
          :npz
        when "safetensors"
          :safetensors
        else
          raise ArgumentError, "unsupported checkpoint format: #{raw.inspect}"
        end
      end

      def __dsl_checkpoint_weights_path(path, checkpoint_format)
        target = path.to_s
        case checkpoint_format
        when :npz
          target.end_with?(".npz") ? target : "#{target}.npz"
        when :safetensors
          target.end_with?(".safetensors") ? target : "#{target}.safetensors"
        else
          target
        end
      end

      def __dsl_checkpoint_sidecar_path(weights_path)
        "#{weights_path}.mlxmeta.json"
      end

      def __dsl_resolve_load_checkpoint_path(path, format)
        target = path.to_s
        return target unless format.nil?
        return target unless File.extname(target).empty?
        return target if File.exist?(target)

        candidates = ["#{target}.npz", "#{target}.safetensors"]
        found = candidates.find { |candidate| File.exist?(candidate) }
        found.nil? ? target : found
      end

      def __dsl_save_native_checkpoint(path, checkpoint_format, optimizer:, metadata:)
        weights_path = __dsl_checkpoint_weights_path(path, checkpoint_format)
        __dsl_ensure_parent_dir!(weights_path)
        save_weights(weights_path)

        payload = {
          "format" => "mlx_dsl_checkpoint_v2_native",
          "weights_format" => checkpoint_format.to_s,
          "metadata" => metadata || {}
        }
        payload["optimizer"] = __dsl_serialize_tree(optimizer.state) unless optimizer.nil?

        File.binwrite(__dsl_checkpoint_sidecar_path(weights_path), JSON.generate(payload))
        weights_path
      end

      def __dsl_ensure_parent_dir!(path)
        dir = File.dirname(path.to_s)
        return if dir.nil? || dir.empty? || dir == "."

        FileUtils.mkdir_p(dir)
      end

      def __dsl_load_native_checkpoint_payload(weights_path)
        sidecar_path = __dsl_checkpoint_sidecar_path(weights_path)
        payload = if File.exist?(sidecar_path)
          JSON.parse(File.binread(sidecar_path))
        else
          {}
        end
        unless payload.is_a?(Hash)
          raise ArgumentError, "invalid native checkpoint sidecar payload"
        end

        payload["format"] ||= "mlx_dsl_checkpoint_v2_native"
        payload["weights_format"] ||= File.extname(weights_path).delete_prefix(".")
        payload["metadata"] ||= {}
        payload
      end

      def __dsl_unknown_kwargs_for_super(kwargs, super_method)
        return [] if kwargs.empty?
        return kwargs.keys unless super_method

        params = super_method.parameters
        return [] if params.any? { |type, _name| type == :keyrest }

        accepted = params.each_with_object(Set.new) do |(type, name), out|
          out << name.to_s if (type == :key || type == :keyreq) && !name.nil?
        end

        kwargs.keys.select { |key| !accepted.include?(key.to_s) }
      end

      def __dsl_extract_declared_options(kwargs)
        out = {}
        option_defs = self.class.dsl_option_definitions
        return out if option_defs.empty? || kwargs.empty?

        option_defs.each_key do |name|
          symbol_key = name.to_sym
          if kwargs.key?(symbol_key)
            out[name] = kwargs.delete(symbol_key)
          elsif kwargs.key?(name)
            out[name] = kwargs.delete(name)
          end
        end
        out
      end

      def __dsl_apply_option_values(provided)
        self.class.dsl_option_definitions.each do |name, spec|
          value = if provided.key?(name)
            provided[name]
          elsif spec[:default].equal?(UNSET)
            if spec[:required]
              raise ArgumentError, "missing required option: #{name}"
            end
          else
            __dsl_resolve_callable(spec[:default])
          end
          next if value.nil? && spec[:default].equal?(UNSET) && !spec[:required]

          public_send("#{name}=", value)
        end
      end

      def __dsl_materialize_declarations!
        return if defined?(@__dsl_materialized) && @__dsl_materialized

        self.class.dsl_declarations.each do |decl|
          case decl[:kind]
          when :layer
            public_send("#{decl[:name]}=", __dsl_build_layer(decl))
          when :param
            public_send("#{decl[:name]}=", __dsl_build_array(decl, default_fill: :uniform))
          when :buffer
            public_send("#{decl[:name]}=", __dsl_build_array(decl, default_fill: :zeros))
            __send__(:no_grad).add(decl[:name])
          else
            raise ArgumentError, "unknown declaration kind: #{decl[:kind]}"
          end
        end

        @__dsl_materialized = true
      end

      def __dsl_build_layer(decl)
        if !decl[:factory].nil?
          value = decl[:factory]
          args = __dsl_resolve_factory_args(decl.fetch(:factory_args, []))
          kwargs = __dsl_resolve_factory_kwargs(decl.fetch(:factory_kwargs, {}))
          if value.is_a?(Class)
            value = value.new(*args, **kwargs)
          elsif value.respond_to?(:call)
            if args.empty? && kwargs.empty?
              value = __dsl_resolve_callable(value)
            elsif kwargs.empty?
              value = value.call(*args)
            else
              value = value.call(*args, **kwargs)
            end
          end
          return __dsl_validate_layer_value(value, decl[:name])
        end

        builder = MLX::DSL::Builder.new(self)
        value = builder.build(&decl[:block])
        if value.nil?
          raise ArgumentError, "layer #{decl[:name]} block returned nil"
        end
        __dsl_validate_layer_value(value, decl[:name])
      end

      def __dsl_validate_layer_value(value, name)
        unless value.is_a?(MLX::NN::Module)
          raise TypeError, "layer #{name} must build an MLX::NN::Module, got #{value.class}"
        end

        value
      end

      def __dsl_build_array(decl, default_fill:)
        shape = __dsl_resolve_shape(decl[:shape])
        dtype = __dsl_resolve_dtype(decl[:dtype])
        init = decl[:init]

        value = if init.nil?
          if default_fill == :uniform
            MLX::Core.random_uniform(shape, -0.05, 0.05, dtype)
          else
            MLX::Core.zeros(shape, dtype)
          end
        else
          __dsl_call_initializer(init, shape, dtype)
        end

        value.is_a?(MLX::Core::Array) ? value : MLX::Core.array(value, dtype)
      end

      def __dsl_resolve_dtype(dtype)
        return __dsl_default_dtype if dtype.equal?(UNSET)

        __dsl_resolve_callable(dtype)
      end

      def __dsl_default_dtype
        if defined?(MLX::Core) && MLX::Core.respond_to?(:float32)
          return MLX::Core.float32
        end

        error_class = if defined?(MLX::Core) && defined?(MLX::Core::NativeUnavailableError)
          MLX::Core::NativeUnavailableError
        else
          RuntimeError
        end
        raise error_class, "MLX native extension is required to initialize DSL params/buffers"
      end

      def __dsl_resolve_shape(shape)
        resolved = __dsl_resolve_callable(shape)
        resolved = [resolved] if resolved.is_a?(Integer)
        unless resolved.is_a?(Array) && resolved.all? { |dim| dim.is_a?(Integer) && dim >= 0 }
          raise ArgumentError, "shape must resolve to an array of non-negative integers"
        end

        resolved
      end

      def __dsl_call_initializer(init, shape, dtype)
        unless init.respond_to?(:call)
          return init
        end

        if init.is_a?(Proc)
          case init.arity
          when 0
            instance_exec(&init)
          when 1
            init.call(shape)
          else
            init.call(shape, dtype)
          end
        else
          init.call(shape, dtype)
        end
      end

      def __dsl_resolve_factory_args(values)
        values.map { |value| __dsl_resolve_callable(value) }
      end

      def __dsl_resolve_factory_kwargs(values)
        values.each_with_object({}) do |(key, value), out|
          out[key] = __dsl_resolve_callable(value)
        end
      end

      def __dsl_resolve_callable(value)
        return value unless value.respond_to?(:call)

        if value.is_a?(Proc)
          if value.arity == 1
            value.call(self)
          else
            instance_exec(&value)
          end
        else
          value.call
        end
      end

      def __dsl_select_paths(matcher)
        path_matcher = __dsl_path_matcher(matcher)
        all_paths = MLX::Utils.tree_flatten(parameters, destination: {}).keys
        all_paths.select { |path| path_matcher.call(path) }
      end

      def __dsl_path_matcher(matcher)
        case matcher
        when Regexp
          lambda { |path| matcher.match?(path) }
        when String, Symbol
          target = matcher.to_s
          lambda { |path| path == target }
        when Array
          targets = matcher.map(&:to_s)
          lambda { |path| targets.include?(path) }
        when Proc
          lambda { |path| matcher.call(path) }
        else
          raise ArgumentError, "unsupported matcher: #{matcher.class}"
        end
      end

      def __dsl_toggle_paths(paths, freeze:)
        module_map = named_modules.to_h
        paths.each do |path|
          module_obj, local_key = __dsl_find_module_for_path(path, module_map)
          next if local_key.nil? || local_key.empty?

          if freeze
            module_obj.__send__(:no_grad).add(local_key)
          else
            module_obj.__send__(:no_grad).delete(local_key)
          end
        end
        self
      end

      def __dsl_serialize_tree(value)
        if value.is_a?(MLX::Core::Array)
          return { "__mlx_array__" => value.__getstate__ }
        end
        if value.is_a?(Array)
          return value.map { |entry| __dsl_serialize_tree(entry) }
        end
        if value.is_a?(Hash)
          return value.each_with_object({}) do |(key, entry), out|
            out[key.to_s] = __dsl_serialize_tree(entry)
          end
        end

        value
      end

      def __dsl_deserialize_tree(value)
        if value.is_a?(Hash) && value.key?("__mlx_array__")
          return __dsl_array_from_state(value.fetch("__mlx_array__"))
        end
        if value.is_a?(Array)
          return value.map { |entry| __dsl_deserialize_tree(entry) }
        end
        if value.is_a?(Hash)
          return value.each_with_object({}) do |(key, entry), out|
            out[key] = __dsl_deserialize_tree(entry)
          end
        end

        value
      end

      def __dsl_array_from_state(state)
        values = state["values"] || state[:values]
        dtype_name = state["dtype"] || state[:dtype]
        if !dtype_name.nil? && MLX::Core.respond_to?(dtype_name.to_sym)
          MLX::Core.array(values, MLX::Core.public_send(dtype_name.to_sym))
        else
          MLX::Core.array(values)
        end
      end

      def __dsl_find_module_for_path(path, module_map)
        best_prefix = ""
        module_map.each_key do |prefix|
          next if prefix.nil? || prefix.empty?
          next if prefix == path
          next unless path.start_with?(prefix + ".")
          next unless prefix.length > best_prefix.length

          best_prefix = prefix
        end

        if best_prefix.empty?
          [self, path]
        else
          [module_map.fetch(best_prefix), path[(best_prefix.length + 1)..]]
        end
      end
    end
  end
end
