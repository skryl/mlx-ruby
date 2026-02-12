# frozen_string_literal: true

module MLX
  module Optimizers
    class Optimizer
      def initialize(learning_rate: 1e-3, schedulers: nil, **_kwargs)
        @initialized = false
        @state = { "step" => 0 }
        @schedulers = {}
        (schedulers || {}).each { |k, v| @schedulers[k.to_s] = v }
        maybe_schedule("learning_rate", learning_rate)
      end

      def update(model, gradients)
        parameters = model.respond_to?(:parameters) ? model.parameters : model
        updated = apply_gradients(gradients, parameters)
        return updated unless model.respond_to?(:update)

        model.update(updated)
      end

      def init(parameters)
        update_state_shape(parameters, @state)
        initialize_parameter_state(parameters, @state)
        @initialized = true
      end

      def init_single(_parameter, state)
        state
      end

      def apply_gradients(gradients, parameters)
        return parameters if gradients.nil? || parameters.nil?

        init(gradients) unless @initialized

        @schedulers.each do |name, scheduler|
          @state[name] = scheduler.call(step)
        end
        @state["step"] = step + 1

        apply_tree(gradients, parameters, @state)
      end

      def apply_single(gradient, parameter, _state = nil)
        if parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)
          MLX::Core.subtract(parameter, MLX::Core.multiply(gradient, learning_rate))
        else
          parameter
        end
      end

      def state
        @state
      end

      def state=(state)
        @initialized = false
        @state = state || {}
      end

      def step
        @state["step"] || 0
      end

      def learning_rate
        @state["learning_rate"]
      end

      def learning_rate=(learning_rate)
        @state["learning_rate"] = learning_rate
      end

      protected

      def maybe_schedule(name, parameter)
        key = name.to_s
        if parameter.respond_to?(:call)
          @schedulers[key] = parameter
          @state[key] = parameter.call(step)
        else
          @state[key] = parameter
        end
      end

      private

      def apply_tree(gradients, parameters, state)
        if gradients.is_a?(Hash) && parameters.is_a?(Hash)
          gradients.each_with_object({}) do |(k, grad), out|
            state_child = if state.is_a?(Hash)
              state[k] ||= {}
            else
              {}
            end
            out[k] = apply_tree(grad, parameters[k], state_child)
          end
        elsif gradients.is_a?(Array) && parameters.is_a?(Array)
          gradients.each_with_index.map do |grad, i|
            state_child = if state.is_a?(Array)
              state[i] ||= {}
            else
              {}
            end
            apply_tree(grad, parameters[i], state_child)
          end
        else
          apply_single(gradients, parameters, state)
        end
      end

      def update_state_shape(parameters, state)
        if parameters.is_a?(Hash)
          state = {} unless state.is_a?(Hash)
          parameters.each do |key, value|
            state[key] = if state.key?(key)
              update_state_shape(value, state[key])
            else
              empty_state_like(value)
            end
          end
          state
        elsif parameters.is_a?(Array)
          current = state.is_a?(Array) ? state : []
          parameters.each_with_index do |value, idx|
            current[idx] = if idx < current.length
              update_state_shape(value, current[idx])
            else
              empty_state_like(value)
            end
          end
          current
        else
          state
        end
      end

      def empty_state_like(parameters)
        if parameters.is_a?(Hash)
          parameters.each_with_object({}) do |(key, value), out|
            out[key] = empty_state_like(value)
          end
        elsif parameters.is_a?(Array)
          parameters.map { |value| empty_state_like(value) }
        else
          {}
        end
      end

      def initialize_parameter_state(parameters, state)
        if parameters.is_a?(Hash)
          parameters.each do |key, value|
            state[key] = initialize_parameter_state(value, state[key])
          end
          state
        elsif parameters.is_a?(Array)
          parameters.each_with_index do |value, idx|
            state[idx] = initialize_parameter_state(value, state[idx])
          end
          state
        else
          state ||= {}
          state = init_single(parameters, state) if state.empty?
          state
        end
      end
    end

    class MultiOptimizer < Optimizer
      def initialize(optimizers, filters: [])
        super(learning_rate: 0.0)
        @state = {}

        if filters.length != optimizers.length - 1
          raise ArgumentError,
                "Given #{filters.length} filters but #{optimizers.length - 1} needed."
        end

        @optimizers = optimizers
        @filters = filters + [lambda { |*_args, **_kwargs| true }]
      end

      def init(parameters)
        @optimizers.zip(split_dictionary(parameters)).each do |optimizer, part|
          optimizer.init(part)
        end
      end

      def apply_gradients(gradients, parameters)
        tree = {}
        @optimizers.zip(split_dictionary(gradients)).each do |optimizer, grads_part|
          tree = MLX::Utils.tree_merge(tree, optimizer.apply_gradients(grads_part, parameters))
        end
        tree
      end

      def state
        { "states" => @optimizers.map(&:state) }
      end

      def state=(state)
        states = if state.is_a?(Hash)
          state["states"] || state[:states]
        end
        if states.nil? || states.length != @optimizers.length
          raise ArgumentError, "Invalid state provided"
        end

        @optimizers.zip(states).each do |optimizer, optimizer_state|
          optimizer.state = optimizer_state
        end
      end

      def learning_rate
        @optimizers.first&.learning_rate
      end

      def learning_rate=(learning_rate)
        @optimizers.each { |optimizer| optimizer.learning_rate = learning_rate }
      end

      private

      def split_dictionary(gradients)
        return [gradients] if @optimizers.length == 1

        parts = Array.new(@optimizers.length) { [] }
        flat_gradients = MLX::Utils.tree_flatten(gradients)
        flat_gradients.each do |path, grad|
          @filters.each_with_index do |fn, i|
            next unless fn.call(path, grad)

            parts[i] << [path, grad]
            break
          end
        end

        parts.map do |part|
          part.empty? ? {} : MLX::Utils.tree_unflatten(part)
        end
      end
    end

    class SGD < Optimizer
      attr_reader :momentum, :weight_decay, :dampening, :nesterov

      def initialize(
        learning_rate: 1e-3,
        momentum: 0.0,
        weight_decay: 0.0,
        dampening: 0.0,
        nesterov: false
      )
        if nesterov && (momentum <= 0 || dampening != 0)
          raise ArgumentError, "Nesterov momentum requires a momentum and zero dampening."
        end

        super(learning_rate: learning_rate)
        @momentum = momentum
        @weight_decay = weight_decay
        @dampening = dampening
        @nesterov = nesterov
      end

      def init_single(parameter, state)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        if weight_decay != 0
          gradient = MLX::Core.add(gradient, MLX::Core.multiply(parameter, weight_decay))
        end

        if momentum <= 0
          return MLX::Core.subtract(parameter, MLX::Core.multiply(gradient, learning_rate))
        end

        velocity = MLX::Core.multiply(state.fetch("v"), momentum)
        if dampening > 0
          velocity = MLX::Core.add(velocity, MLX::Core.multiply(gradient, 1 - dampening))
        else
          velocity = MLX::Core.add(velocity, gradient)
        end

        update = if nesterov
          MLX::Core.add(gradient, MLX::Core.multiply(velocity, momentum))
        else
          velocity
        end

        state["v"] = velocity
        MLX::Core.subtract(parameter, MLX::Core.multiply(update, learning_rate))
      end
    end
    class RMSprop < Optimizer
      attr_reader :alpha, :eps

      def initialize(learning_rate: 1e-3, alpha: 0.99, eps: 1e-8)
        super(learning_rate: learning_rate)
        @alpha = alpha
        @eps = eps

        if @alpha < 0.0
          raise ArgumentError, "RMSprop alpha should be >=0, #{@alpha} was provided instead"
        end
        if @eps < 0.0
          raise ArgumentError, "RMSprop epsilon should be >0, #{@eps} was provided instead"
        end
      end

      def init_single(parameter, state)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        velocity = state.fetch("v")
        velocity = MLX::Core.add(
          MLX::Core.multiply(velocity, alpha),
          MLX::Core.multiply(MLX::Core.square(gradient), 1 - alpha)
        )
        state["v"] = velocity

        denom = MLX::Core.add(MLX::Core.sqrt(velocity), eps)
        step_update = MLX::Core.divide(MLX::Core.multiply(gradient, learning_rate), denom)
        MLX::Core.subtract(parameter, step_update)
      end
    end
    class Adagrad < Optimizer
      attr_reader :eps

      def initialize(learning_rate: 1e-3, eps: 1e-8)
        super(learning_rate: learning_rate)
        @eps = eps
        if @eps < 0.0
          raise ArgumentError, "Adagrad epsilon should be >0, #{@eps} was provided instead"
        end
      end

      def init_single(parameter, state)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        velocity = MLX::Core.add(state.fetch("v"), MLX::Core.square(gradient))
        state["v"] = velocity

        denom = MLX::Core.add(MLX::Core.sqrt(velocity), eps)
        step_update = MLX::Core.divide(MLX::Core.multiply(gradient, learning_rate), denom)
        MLX::Core.subtract(parameter, step_update)
      end
    end

    class AdaDelta < Optimizer
      attr_reader :rho, :eps

      def initialize(learning_rate: 1e-3, rho: 0.9, eps: 1e-6)
        super(learning_rate: learning_rate)
        @rho = rho
        @eps = eps

        if @rho < 0.0
          raise ArgumentError, "AdaDelta rho should be >=0, #{@rho} was provided instead"
        end
        if @eps < 0.0
          raise ArgumentError, "AdaDelta epsilon should be >0, #{@eps} was provided instead"
        end
      end

      def init_single(parameter, state)
        state["v"] = MLX::Core.zeros_like(parameter)
        state["u"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        velocity = state.fetch("v")
        update_acc = state.fetch("u")

        velocity = MLX::Core.add(
          MLX::Core.multiply(velocity, rho),
          MLX::Core.multiply(MLX::Core.square(gradient), 1 - rho)
        )
        delta = MLX::Core.multiply(
          MLX::Core.divide(
            MLX::Core.sqrt(MLX::Core.add(update_acc, eps)),
            MLX::Core.sqrt(MLX::Core.add(velocity, eps))
          ),
          gradient
        )
        update_acc = MLX::Core.add(
          MLX::Core.multiply(update_acc, rho),
          MLX::Core.multiply(MLX::Core.square(delta), 1 - rho)
        )

        state["v"] = velocity
        state["u"] = update_acc

        MLX::Core.subtract(parameter, MLX::Core.multiply(delta, learning_rate))
      end
    end
    class Adam < Optimizer
      attr_reader :betas, :eps, :bias_correction

      def initialize(learning_rate: 1e-3, betas: [0.9, 0.999], eps: 1e-8, bias_correction: false)
        super(learning_rate: learning_rate)
        @betas = betas
        @eps = eps
        @bias_correction = bias_correction
      end

      def init_single(parameter, state)
        state["m"] = MLX::Core.zeros_like(parameter)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        b1, b2 = betas
        m = state.fetch("m")
        v = state.fetch("v")
        m = MLX::Core.add(MLX::Core.multiply(m, b1), MLX::Core.multiply(gradient, 1 - b1))
        v = MLX::Core.add(MLX::Core.multiply(v, b2), MLX::Core.multiply(MLX::Core.square(gradient), 1 - b2))
        state["m"] = m
        state["v"] = v

        if bias_correction
          c1 = learning_rate.to_f / (1 - (b1**step))
          c2 = 1.0 / Math.sqrt(1 - (b2**step))
          numerator = MLX::Core.multiply(m, c1)
          denominator = MLX::Core.add(MLX::Core.multiply(MLX::Core.sqrt(v), c2), eps)
          MLX::Core.subtract(parameter, MLX::Core.divide(numerator, denominator))
        else
          numerator = MLX::Core.multiply(m, learning_rate)
          denominator = MLX::Core.add(MLX::Core.sqrt(v), eps)
          MLX::Core.subtract(parameter, MLX::Core.divide(numerator, denominator))
        end
      end
    end
    class AdamW < Adam
      attr_reader :weight_decay

      def initialize(
        learning_rate: 1e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: 0.01,
        bias_correction: false
      )
        super(
          learning_rate: learning_rate,
          betas: betas,
          eps: eps,
          bias_correction: bias_correction
        )
        @weight_decay = weight_decay
      end

      def apply_single(gradient, parameter, state)
        lr = learning_rate.to_f
        decayed_parameter = MLX::Core.multiply(parameter, 1 - lr * weight_decay)
        super(gradient, decayed_parameter, state)
      end
    end

    class Adamax < Adam
      def initialize(learning_rate: 1e-3, betas: [0.9, 0.999], eps: 1e-8)
        super(learning_rate: learning_rate, betas: betas, eps: eps, bias_correction: false)
        if eps < 0.0
          raise ArgumentError, "Epsilon value should be >=0, #{eps} was provided instead"
        end
      end

      def init_single(parameter, state)
        state["m"] = MLX::Core.zeros_like(parameter)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        b1, b2 = betas
        m = state.fetch("m")
        v = state.fetch("v")

        m = MLX::Core.add(MLX::Core.multiply(m, b1), MLX::Core.multiply(gradient, 1 - b1))
        v = MLX::Core.maximum(MLX::Core.multiply(v, b2), MLX::Core.abs(gradient))
        state["m"] = m
        state["v"] = v

        numerator = MLX::Core.multiply(m, learning_rate)
        denominator = MLX::Core.add(v, eps)
        MLX::Core.subtract(parameter, MLX::Core.divide(numerator, denominator))
      end
    end
    class Lion < Optimizer
      attr_reader :betas, :weight_decay

      def initialize(learning_rate: 1e-3, betas: [0.9, 0.99], weight_decay: 0.0)
        super(learning_rate: learning_rate)
        @betas = betas
        @weight_decay = weight_decay
      end

      def init_single(parameter, state)
        state["m"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        b1, b2 = betas
        momentum = state.fetch("m")
        c = MLX::Core.add(
          MLX::Core.multiply(momentum, b1),
          MLX::Core.multiply(gradient, 1 - b1)
        )
        state["m"] = MLX::Core.add(
          MLX::Core.multiply(momentum, b2),
          MLX::Core.multiply(gradient, 1 - b2)
        )

        updated_parameter = parameter
        if weight_decay > 0
          updated_parameter = MLX::Core.multiply(updated_parameter, 1 - learning_rate.to_f * weight_decay)
        end

        MLX::Core.subtract(updated_parameter, MLX::Core.multiply(MLX::Core.sign(c), learning_rate))
      end
    end
    class Adafactor < Optimizer
      attr_reader :eps, :clip_threshold, :decay_rate, :beta_1, :weight_decay, :scale_parameter,
                  :relative_step, :warmup_init

      def initialize(
        learning_rate: nil,
        eps: [1e-30, 1e-3],
        clip_threshold: 1.0,
        decay_rate: -0.8,
        beta_1: nil,
        weight_decay: 0.0,
        scale_parameter: true,
        relative_step: true,
        warmup_init: false
      )
        super(learning_rate: (learning_rate.nil? ? 1e-3 : learning_rate))
        @eps = eps
        @clip_threshold = clip_threshold
        @decay_rate = decay_rate
        @beta_1 = beta_1
        @weight_decay = weight_decay
        @scale_parameter = scale_parameter
        @relative_step = relative_step
        @warmup_init = warmup_init
      end

      def init_single(parameter, state)
        if parameter.ndim >= 2
          shape = parameter.shape
          dtype = parameter.dtype
          state["exp_avg_sq_row"] = MLX::Core.zeros(shape[0...-1], dtype)
          state["exp_avg_sq_col"] = MLX::Core.zeros(shape[0...-2] + shape[-1..], dtype)
        else
          state["exp_avg_sq"] = MLX::Core.zeros_like(parameter)
        end

        state["exp_avg"] = MLX::Core.zeros_like(parameter) unless beta_1.nil?
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        factored = gradient.ndim >= 2
        current_step = step.to_f
        use_first_moment = !beta_1.nil?

        parameter_rms = compute_rms(parameter)
        lr = compute_learning_rate(current_step, parameter_rms)
        beta_2 = 1.0 - (current_step**decay_rate)
        update = MLX::Core.add(MLX::Core.square(gradient), eps[0])

        if factored
          exp_avg_sq_row = state.fetch("exp_avg_sq_row")
          exp_avg_sq_col = state.fetch("exp_avg_sq_col")
          exp_avg_sq_row = MLX::Core.add(
            MLX::Core.multiply(exp_avg_sq_row, beta_2),
            MLX::Core.multiply(MLX::Core.mean(update, -1), 1 - beta_2)
          )
          exp_avg_sq_col = MLX::Core.add(
            MLX::Core.multiply(exp_avg_sq_col, beta_2),
            MLX::Core.multiply(MLX::Core.mean(update, -2), 1 - beta_2)
          )
          state["exp_avg_sq_row"] = exp_avg_sq_row
          state["exp_avg_sq_col"] = exp_avg_sq_col
          update = MLX::Core.multiply(approximate_exp_moving_avg(exp_avg_sq_row, exp_avg_sq_col), gradient)
        else
          exp_avg_sq = state.fetch("exp_avg_sq")
          exp_avg_sq = MLX::Core.add(
            MLX::Core.multiply(exp_avg_sq, beta_2),
            MLX::Core.multiply(update, 1 - beta_2)
          )
          state["exp_avg_sq"] = exp_avg_sq
          update = MLX::Core.multiply(MLX::Core.rsqrt(exp_avg_sq), gradient)
        end

        rms_update = scalar(compute_rms(update))
        normalizer = [1.0, rms_update / clip_threshold.to_f].max
        update = MLX::Core.divide(update, normalizer)
        update = MLX::Core.multiply(update, lr)

        if use_first_moment
          exp_avg = state.fetch("exp_avg")
          exp_avg = MLX::Core.add(
            MLX::Core.multiply(exp_avg, beta_1.to_f),
            MLX::Core.multiply(update, 1 - beta_1.to_f)
          )
          state["exp_avg"] = exp_avg
          update = exp_avg
        end

        if weight_decay != 0
          parameter = MLX::Core.add(parameter, MLX::Core.multiply(parameter, -weight_decay.to_f * lr))
        end

        MLX::Core.subtract(parameter, update)
      end

      private

      def compute_rms(inputs)
        MLX::Core.sqrt(MLX::Core.mean(MLX::Core.square(inputs)))
      end

      def compute_learning_rate(current_step, parameter_rms)
        if relative_step
          min_step = warmup_init ? 1e-6 * current_step : 1e-2
          relative_step_size = [min_step, 1.0 / Math.sqrt(current_step)].min
        else
          relative_step_size = learning_rate.to_f
        end

        parameter_scale = if scale_parameter
          [eps[1].to_f, scalar(parameter_rms)].max
        else
          1.0
        end
        parameter_scale * relative_step_size
      end

      def approximate_exp_moving_avg(exp_avg_sq_row, exp_avg_sq_col)
        mean_row = if exp_avg_sq_row.ndim > 1
          MLX::Core.expand_dims(MLX::Core.mean(exp_avg_sq_row, -1), -1)
        else
          MLX::Core.mean(exp_avg_sq_row)
        end
        r_factor = MLX::Core.rsqrt(MLX::Core.divide(exp_avg_sq_row, mean_row))
        c_factor = MLX::Core.rsqrt(exp_avg_sq_col)
        MLX::Core.matmul(MLX::Core.expand_dims(r_factor, -1), MLX::Core.expand_dims(c_factor, 0))
      end

      def scalar(value)
        if value.respond_to?(:item)
          value.item.to_f
        else
          value.to_f
        end
      end
    end
    class Muon < Optimizer
      attr_reader :momentum, :weight_decay, :nesterov, :ns_steps

      def initialize(
        learning_rate: 1e-3,
        momentum: 0.95,
        weight_decay: 0.01,
        nesterov: true,
        ns_steps: 5
      )
        super(learning_rate: learning_rate)
        @momentum = momentum
        @weight_decay = weight_decay
        @nesterov = nesterov
        @ns_steps = ns_steps
      end

      def init_single(parameter, state)
        state["v"] = MLX::Core.zeros_like(parameter)
        state
      end

      def apply_single(gradient, parameter, state)
        return parameter unless parameter.is_a?(MLX::Core::Array) && gradient.is_a?(MLX::Core::Array)

        if weight_decay != 0
          gradient = MLX::Core.add(gradient, MLX::Core.multiply(parameter, weight_decay))
        end

        velocity = MLX::Core.add(
          MLX::Core.multiply(state.fetch("v"), momentum),
          MLX::Core.multiply(gradient, 1 - momentum)
        )
        state["v"] = velocity

        update = if nesterov
          MLX::Core.add(
            MLX::Core.multiply(gradient, 1 - momentum),
            MLX::Core.multiply(velocity, momentum)
          )
        else
          velocity
        end

        lr = learning_rate.to_f
        if update.ndim >= 2
          original_shape = update.shape
          reshape_needed = update.ndim > 2

          if reshape_needed
            rest = original_shape[1..].reduce(1) { |acc, d| acc * d }
            update = MLX::Core.reshape(update, [original_shape[0], rest])
          end

          update = zeropower_via_newtonschulz5(update, steps: ns_steps)
          update = MLX::Core.reshape(update, original_shape) if reshape_needed

          ratio = update.shape[-2].to_f / update.shape[-1].to_f
          lr *= Math.sqrt([1.0, ratio].max)
        end

        MLX::Core.subtract(parameter, MLX::Core.multiply(update, lr))
      end

      private

      def zeropower_via_newtonschulz5(x, steps:)
        unless x.ndim == 2
          raise ArgumentError, "Expected a 2D array for Newton-Schulz iteration, got shape #{x.shape} instead."
        end

        a, b, c = 3.4445, -4.7750, 2.0315
        transpose_needed = x.shape[-2] > x.shape[-1]
        x = x.T if transpose_needed

        x = MLX::Core.divide(x, MLX::Core.add(MLX::Core.norm(x), 1e-7))
        steps.to_i.times do
          a_mat = MLX::Core.matmul(x, x.T)
          b_mat = MLX::Core.addmm(MLX::Core.multiply(a_mat, b), a_mat, a_mat, 1.0, c)
          x = MLX::Core.addmm(MLX::Core.multiply(x, a), b_mat, x, 1.0, 1.0)
        end

        x = x.T if transpose_needed
        x
      end
    end

    def self.clip_grad_norm(grads, max_norm)
      flatten = lambda do |value|
        if value.is_a?(Array)
          value.flat_map { |item| flatten.call(item) }
        else
          [value.to_f]
        end
      end

      norm_squared = MLX::Utils.tree_flatten(grads).reduce(0.0) do |acc, (_k, grad)|
        values = flatten.call(grad.to_a)
        acc + values.reduce(0.0) { |sum, v| sum + (v * v) }
      end
      total_norm = Math.sqrt(norm_squared)
      normalizer = [max_norm.to_f / (total_norm + 1e-6), 1.0].min

      clipped = MLX::Utils.tree_map(lambda { |g| MLX::Core.multiply(g, normalizer) }, grads)
      [clipped, MLX::Core.array(total_norm, MLX::Core.float32)]
    end
  end
end
