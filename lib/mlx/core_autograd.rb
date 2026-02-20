# frozen_string_literal: true

module MLX
  module Core
    module AutogradInternals
      private

      def normalize_diff_targets(argnums, argnames)
        argnames_v = normalize_argnames(argnames)
        argnums_v = normalize_argnums(argnums, argnames_v)
        if argnums_v.empty? && argnames_v.empty?
          raise ArgumentError, "Gradient wrt no argument requested"
        end
        [argnums_v, argnames_v]
      end

      def normalize_argnums(argnums, argnames)
        if argnums.nil?
          return argnames.empty? ? [0] : []
        end
        values = if argnums.is_a?(::Integer)
          [argnums]
        elsif argnums.is_a?(::Array)
          argnums
        else
          raise TypeError, "argnums must be an Integer, an Array of Integer, or nil"
        end
        out = values.map do |value|
          raise TypeError, "argnums entries must be Integer" unless value.is_a?(::Integer)
          raise ArgumentError, "argnums cannot contain negative values" if value.negative?
          value
        end
        raise ArgumentError, "duplicate argnums are not allowed" if out.uniq.length != out.length

        out
      end

      def normalize_argnames(argnames)
        return [] if argnames.nil?
        values = if argnames.is_a?(::String) || argnames.is_a?(::Symbol)
          [argnames]
        elsif argnames.is_a?(::Array)
          argnames
        else
          raise TypeError, "argnames must be a String, Symbol, Array, or nil"
        end
        out = values.map(&:to_s)
        raise ArgumentError, "duplicate argnames are not allowed" if out.uniq.length != out.length

        out
      end

      def build_grad_like_function(fun, argnums, argnames, with_value)
        cache = {}

        lambda do |*args, **kwargs|
          selections, flat_inputs = build_target_selections(args, kwargs, argnums, argnames)
          cache_key = grad_selection_cache_key(selections)
          entry = cache[cache_key]
          unless entry
            call_state = { mutex: Mutex.new, stacks: {} }
            lifted = lambda do |*flat_vars|
              state = grad_call_state_current(call_state)
              if state.nil?
                raise RuntimeError, "gradient transform invoked without call state"
              end

              call_args, call_kwargs = apply_flat_vars_to_targets(
                state[:args],
                state[:kwargs],
                state[:selections],
                flat_vars
              )
              raw_value = fun.call(*call_args, **call_kwargs)
              state[:captured_value] = raw_value
              extract_loss(raw_value)
            end

            native_argnums = (0...flat_inputs.length).to_a
            native_fn = if with_value
              native_value_and_grad(lifted, native_argnums)
            else
              native_grad(lifted, native_argnums)
            end

            entry = {
              native_fn: native_fn,
              call_state: call_state
            }
            cache[cache_key] = entry
          end

          state = {
            args: args,
            kwargs: kwargs,
            selections: selections,
            captured_value: nil
          }
          grad_call_state_push(entry[:call_state], state)

          if with_value
            _loss, raw_grads = entry[:native_fn].call(*flat_inputs)
            value = state[:captured_value]
            value = fun.call(*args, **kwargs) if value.nil?
            [value, rebuild_grad_result(raw_grads, selections, argnames)]
          else
            raw_grads = entry[:native_fn].call(*flat_inputs)
            rebuild_grad_result(raw_grads, selections, argnames)
          end
        ensure
          grad_call_state_pop(entry[:call_state]) unless entry.nil?
        end
      end

      def grad_call_state_current(call_state)
        thread = Thread.current
        call_state[:mutex].synchronize do
          stack = call_state[:stacks][thread]
          stack&.last
        end
      end

      def grad_call_state_push(call_state, state)
        thread = Thread.current
        call_state[:mutex].synchronize do
          stack = call_state[:stacks][thread]
          if stack.nil?
            stack = []
            call_state[:stacks][thread] = stack
          end
          stack << state
        end
      end

      def grad_call_state_pop(call_state)
        thread = Thread.current
        call_state[:mutex].synchronize do
          stack = call_state[:stacks][thread]
          return if stack.nil?

          stack.pop
          call_state[:stacks].delete(thread) if stack.empty?
        end
      end

      def build_custom_vjp_grad_function(fun)
        lambda do |*args, **kwargs|
          unless kwargs.empty?
            raise ArgumentError, "custom-function grad currently supports positional arguments only"
          end
          outputs = normalize_array_output(fun.call(*args), "custom_function output")
          cotangents = outputs.map { |out| MLX::Core.ones_like(out) }
          output_arg = outputs.length == 1 ? outputs[0] : outputs
          grads = normalize_array_output(
            fun.call_custom_vjp(args, cotangents, output_arg),
            "custom_function vjp output"
          )
          grads.length == 1 ? grads[0] : grads
        end
      end

      def build_custom_vjp_value_and_grad_function(fun)
        grad_fn = build_custom_vjp_grad_function(fun)
        lambda do |*args, **kwargs|
          value = fun.call(*args, **kwargs)
          [value, grad_fn.call(*args, **kwargs)]
        end
      end

      def extract_loss(output)
        return output if output.is_a?(MLX::Core::Array)

        if output.is_a?(::Array) && !output.empty? && output[0].is_a?(MLX::Core::Array)
          return output[0]
        end

        raise ArgumentError,
              "function must return an MLX::Core::Array or an Array whose first element is an MLX::Core::Array"
      end

      def build_target_selections(args, kwargs, argnums, argnames)
        positional = []
        keyword = []
        flat_inputs = []

        argnums.each do |index|
          if index >= args.length
            raise ArgumentError,
                  "Can't compute gradient for positional argument #{index} when #{args.length} positional arguments were provided"
          end
          spec = flatten_tree_spec(args[index], flat_inputs, true)
          positional << { index: index, spec: spec }
        end

        argnames.each do |name|
          key = kwarg_key_for_name(kwargs, name)
          unless key
            raise ArgumentError,
                  "Can't compute gradient for keyword argument '#{name}' because it was not provided"
          end
          spec = flatten_tree_spec(kwargs[key], flat_inputs, true)
          keyword << { key: key, name: name, spec: spec }
        end

        [{ positional: positional, keyword: keyword }, flat_inputs]
      end

      def grad_selection_cache_key(selections)
        positional = selections[:positional].map do |entry|
          "#{entry[:index]}:#{structure_cache_key(entry[:spec])}"
        end
        keyword = selections[:keyword].map do |entry|
          "#{entry[:name]}:#{entry[:key]}:#{structure_cache_key(entry[:spec])}"
        end
        "P[#{positional.join(',')}]K[#{keyword.join(',')}]"
      end

      def normalize_raw_grads(raw)
        normalize_array_sequence(raw, "gradient")
      end

      def rebuild_grad_result(raw_grads, selections, argnames)
        grad_arrays = normalize_raw_grads(raw_grads)
        cursor = 0

        positional_grads = selections[:positional].map do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], grad_arrays, cursor)
          value
        end
        keyword_grads = {}
        selections[:keyword].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], grad_arrays, cursor)
          keyword_grads[entry[:name]] = value
        end
        unless cursor == grad_arrays.length
          raise RuntimeError, "internal gradient reconstruction mismatch"
        end

        if argnames.empty?
          return positional_grads[0] if positional_grads.length == 1
          return positional_grads
        end

        positional_out = if positional_grads.empty?
          nil
        elsif positional_grads.length == 1
          positional_grads[0]
        else
          positional_grads
        end
        [positional_out, keyword_grads]
      end

      def apply_flat_vars_to_targets(args, kwargs, selections, flat_vars)
        rebuilt_args = args.dup
        rebuilt_kwargs = kwargs.dup
        cursor = 0

        selections[:positional].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], flat_vars, cursor)
          rebuilt_args[entry[:index]] = value
        end

        selections[:keyword].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], flat_vars, cursor)
          rebuilt_kwargs[entry[:key]] = value
        end

        unless cursor == flat_vars.length
          raise RuntimeError, "internal target reconstruction mismatch"
        end
        [rebuilt_args, rebuilt_kwargs]
      end

      def kwarg_key_for_name(kwargs, name)
        symbol = name.to_sym
        return symbol if kwargs.key?(symbol)
        return name if kwargs.key?(name)

        nil
      end
    end
  end
end
