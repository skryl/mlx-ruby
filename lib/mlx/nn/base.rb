# frozen_string_literal: true

require "set"

module MLX
  module NN
    class Module
      def initialize
        @training = true
        @no_grad = Set.new
        @state = {}
      end

      def state
        @state
      end

      def load_weights(file_or_weights, strict: true)
        weights = file_or_weights
        if weights.is_a?(String)
          loaded = MLX::Core.load(weights)
          weights = loaded.to_a
        elsif weights.is_a?(Hash)
          weights = weights.to_a
        end

        normalized_weights = weights.map { |k, v| [k.to_s, v] }
        if strict
          new_weights = normalized_weights.to_h
          current_weights = MLX::Utils.tree_flatten(parameters, destination: {})
          extras = new_weights.keys - current_weights.keys
          unless extras.empty?
            raise ArgumentError, "Received #{extras.length} parameters not in model: \n#{extras.sort.join(",\n")}."
          end

          missing = current_weights.keys - new_weights.keys
          unless missing.empty?
            raise ArgumentError, "Missing #{missing.length} parameters: \n#{missing.sort.join(",\n")}."
          end

          current_weights.each do |key, value|
            new_value = new_weights[key]
            unless new_value.is_a?(MLX::Core::Array)
              raise ArgumentError, "Expected MLX::Core::Array but received #{new_value.class} for parameter #{key}"
            end
            if new_value.shape != value.shape
              raise ArgumentError,
                    "Expected shape #{value.shape} but received shape #{new_value.shape} for parameter #{key}"
            end
          end
        end

        update(MLX::Utils.tree_unflatten(normalized_weights), strict: false) unless normalized_weights.empty?
        self
      end

      def save_weights(file)
        params_dict = MLX::Utils.tree_flatten(parameters, destination: {})

        if file.end_with?(".npz")
          kwargs = params_dict.each_with_object({}) do |(k, v), out|
            out[k.to_sym] = v
          end
          MLX::Core.savez(file, **kwargs)
        elsif file.end_with?(".safetensors")
          MLX::Core.save_safetensors(file, params_dict)
        else
          raise ArgumentError, "Unsupported file extension for #{file}. Use '.npz' or '.safetensors'."
        end
      end

      def parameters
        filter_and_map(method(:valid_parameter_filter))
      end

      def trainable_parameters
        filter_and_map(method(:trainable_parameter_filter))
      end

      def children
        filter_and_map(
          method(:valid_child_filter),
          is_leaf_fn: lambda do |_module, _key, value|
            value.is_a?(Module)
          end
        )
      end

      def leaf_modules
        filter_and_map(
          method(:valid_child_filter),
          is_leaf_fn: lambda do |_module, _key, value|
            value.is_a?(Module) && MLX::Utils.tree_flatten(value.children).empty?
          end
        )
      end

      def modules
        module_list = []
        apply_to_modules { |_key, mod| module_list << mod }
        module_list
      end

      def named_modules
        module_list = []
        apply_to_modules { |key, mod| module_list << [key, mod] }
        module_list
      end

      def update(parameters = {}, strict: true)
        apply_update(@state, parameters, strict)
        self
      end

      def apply(fn = nil, &block)
        callback = fn || block
        if callback
          update(filter_and_map(method(:valid_parameter_filter), callback))
        end
        self
      end

      def update_modules(modules = {}, strict: true)
        update_modules_impl(@state, modules, strict)
        self
      end

      def apply_to_modules(fn = nil, &block)
        callback = fn || block
        return self unless callback

        module_stack = [["", self]]
        until module_stack.empty?
          prefix, mod = module_stack.pop
          callback.call(prefix, mod)
          MLX::Utils.tree_flatten(mod.children).each do |path, child|
            next unless child.is_a?(Module)

            child_prefix = prefix.empty? ? path : "#{prefix}.#{path}"
            module_stack << [child_prefix, child]
          end
        end
        self
      end

      def freeze(recurse: true, keys: nil, strict: false)
        freeze_impl = lambda do |_name, mod|
          local_keys = keys
          if local_keys.nil?
            flat = MLX::Utils.tree_flatten(
              mod.__send__(
                :filter_and_map,
                lambda do |m, key, value|
                  !value.is_a?(Module) && m.__send__(:valid_parameter_filter, m, key, value)
                end
              )
            )
            local_keys = flat.map { |key, _| key }
          end

          local_keys = mod.__send__(:validate_keys, local_keys, strict)
          local_keys.each { |key| mod.__send__(:no_grad).add(key) }
        end

        if recurse
          apply_to_modules(&freeze_impl)
        else
          freeze_impl.call("", self)
        end
        self
      end

      def unfreeze(recurse: true, keys: nil, strict: false)
        unfreeze_impl = lambda do |_name, mod|
          if keys.nil?
            mod.__send__(:no_grad).clear
          else
            local_keys = mod.__send__(:validate_keys, keys, strict)
            local_keys.each { |key| mod.__send__(:no_grad).delete(key) }
          end
        end

        if recurse
          apply_to_modules(&unfreeze_impl)
        else
          unfreeze_impl.call("", self)
        end
        self
      end

      def train(mode = true)
        @training = !!mode
        self
      end

      def eval
        @training = false
        self
      end

      def set_dtype(*_args, **_kwargs)
        self
      end

      def method_missing(name, *args, &block)
        method_name = name.to_s
        if method_name.end_with?("=")
          key = method_name[0...-1]
          return assign_member(key, args.first)
        end

        if args.empty? && block.nil?
          return @state[method_name] if @state.key?(method_name)

          ivar = :"@#{method_name}"
          return instance_variable_get(ivar) if instance_variable_defined?(ivar)
        end

        super
      end

      def respond_to_missing?(name, include_private = false)
        method_name = name.to_s
        return true if method_name.end_with?("=")
        return true if @state.key?(method_name)

        ivar = :"@#{method_name}"
        return true if instance_variable_defined?(ivar)

        super
      end

      def call(*_args, **_kwargs)
        raise NotImplementedError, "#{self.class}#call is not implemented"
      end

      private

      attr_reader :no_grad

      def assign_member(key, value)
        if storable_member?(value)
          @state[key] = value
          ivar = :"@#{key}"
          remove_instance_variable(ivar) if instance_variable_defined?(ivar)
        else
          instance_variable_set(:"@#{key}", value)
          @state.delete(key)
        end
        value
      end

      def storable_member?(value)
        value.is_a?(MLX::Core::Array) || value.is_a?(Hash) || value.is_a?(Array) || value.is_a?(Module)
      end

      def valid_child_filter(_module, _key, value)
        value.is_a?(Module) || value.is_a?(Hash) || value.is_a?(Array)
      end

      def valid_parameter_filter(_module, key, value)
        (value.is_a?(Module) || value.is_a?(Hash) || value.is_a?(Array) || value.is_a?(MLX::Core::Array)) &&
          !key.to_s.start_with?("_")
      end

      def trainable_parameter_filter(module_obj, key, value)
        valid_parameter_filter(module_obj, key, value) && !module_obj.no_grad.include?(key.to_s)
      end

      def filter_and_map(filter_fn, map_fn = nil, is_leaf_fn: nil)
        map_fn ||= lambda { |x| x }
        is_leaf_fn ||= lambda do |_module, _key, value|
          !(value.is_a?(Module) || value.is_a?(Hash) || value.is_a?(Array))
        end

        @state.each_with_object({}) do |(key, value), out|
          next unless filter_fn.call(self, key, value)

          out[key] = unwrap(self, key, value, filter_fn, map_fn, is_leaf_fn)
        end
      end

      def unwrap(model_obj, value_key, value, filter_fn, map_fn, is_leaf_fn)
        if is_leaf_fn.call(model_obj, value_key, value)
          map_fn.call(value)
        elsif value.is_a?(Module)
          value.state.each_with_object({}) do |(k, v), out|
            next unless filter_fn.call(value, k, v)

            out[k] = unwrap(value, k, v, filter_fn, map_fn, is_leaf_fn)
          end
        elsif value.is_a?(Hash)
          value.each_with_object({}) do |(k, v), out|
            tk = "#{value_key}.#{k}"
            out[k] = filter_fn.call(model_obj, tk, v) ? unwrap(model_obj, tk, v, filter_fn, map_fn, is_leaf_fn) : {}
          end
        elsif value.is_a?(Array)
          value.each_with_index.map do |v, i|
            tk = "#{value_key}.#{i}"
            filter_fn.call(model_obj, tk, v) ? unwrap(model_obj, tk, v, filter_fn, map_fn, is_leaf_fn) : {}
          end
        else
          raise RuntimeError, "Unexpected leaf found while traversing the module"
        end
      end

      def validate_keys(keys, strict)
        key_list = keys.is_a?(Array) ? keys : [keys]
        key_list = key_list.map(&:to_s)
        if strict
          key_list.each do |key|
            raise KeyError, "Module doesn't contain member #{key}." unless @state.key?(key)
          end
        end
        key_list
      end

      def update_modules_impl(dst, modules, strict)
        dst = dst.state if dst.is_a?(Module)
        if modules.is_a?(Hash)
          modules.each do |k, new_value|
            if dst.key?(k)
              current_value = dst[k]
              if current_value.is_a?(Module) && new_value.is_a?(Module)
                dst[k] = new_value
              elsif current_value.is_a?(Hash) || current_value.is_a?(Array)
                update_modules_impl(current_value, new_value, strict)
              elsif strict && new_value != {}
                raise ArgumentError, "Received invalid type: #{new_value.class}."
              end
            elsif strict
              raise ArgumentError, "Module does not have sub-module named \"#{k}\"."
            end
          end
        elsif modules.is_a?(Array) && dst.is_a?(Array)
          modules.each_with_index do |new_value, i|
            current_value = dst[i]
            if current_value.is_a?(Module) && new_value.is_a?(Module)
              dst[i] = new_value
            elsif current_value.is_a?(Hash) || current_value.is_a?(Array)
              update_modules_impl(current_value, new_value, strict)
            elsif strict && new_value != {}
              raise ArgumentError, "Received invalid type: #{new_value.class}."
            end
          end
        elsif strict
          raise ArgumentError, "Received invalid type: #{modules.class}."
        end
      end

      def apply_update(dst, parameters, strict)
        dst = dst.state if dst.is_a?(Module)

        if parameters.is_a?(Hash)
          parameters.each do |key, new_value|
            if dst.key?(key)
              current_value = dst[key]
              if current_value.is_a?(MLX::Core::Array)
                if strict && !new_value.is_a?(MLX::Core::Array)
                  raise ArgumentError, "Received invalid type: #{new_value.class}."
                end
                dst[key] = new_value
              else
                apply_update(current_value, new_value, strict)
              end
            elsif strict
              raise ArgumentError, "Module does not have parameter named \"#{key}\"."
            end
          end
        elsif parameters.is_a?(Array) && dst.is_a?(Array)
          parameters.each_with_index do |new_value, i|
            current_value = dst[i]
            if current_value.is_a?(MLX::Core::Array)
              if strict && !new_value.is_a?(MLX::Core::Array)
                raise ArgumentError, "Received invalid type: #{new_value.class}."
              end
              dst[i] = new_value
            else
              apply_update(current_value, new_value, strict)
            end
          end
        elsif strict
          raise ArgumentError, "Received invalid type: #{parameters.class}."
        end
      end
    end
  end
end
