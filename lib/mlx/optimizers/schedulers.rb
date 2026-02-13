# frozen_string_literal: true

module MLX
  module Optimizers
    module Schedulers
      module_function

      def exponential_decay(init, decay_rate)
        lambda { |step| init * (decay_rate**step) }
      end

      def step_decay(init, decay_rate, step_size)
        lambda do |step|
          power = (step / step_size).floor
          init * (decay_rate**power)
        end
      end

      def cosine_decay(init, decay_steps, end_value = 0.0)
        lambda do |step|
          bounded_step = [step.to_f, decay_steps.to_f].min
          ratio = bounded_step / decay_steps.to_f
          end_value + 0.5 * (init - end_value) * (1.0 + Math.cos(Math::PI * ratio))
        end
      end

      def join_schedules(schedules, boundaries)
        if schedules.empty?
          raise ArgumentError, "Must provide at least 1 schedule to join."
        end

        if schedules.length != boundaries.length + 1
          raise ArgumentError,
                "Received #{boundaries.length} boundaries but expected #{schedules.length - 1}."
        end

        lambda do |step|
          output = schedules[0].call(step)
          boundaries.each_with_index do |boundary, idx|
            output = schedules[idx + 1].call(step - boundary) if step >= boundary
          end
          output
        end
      end

      def linear_schedule(init, end_value, steps)
        raise ArgumentError, "steps must be greater than 0, but got #{steps}." if steps < 1

        lambda do |step|
          bounded_step = [step.to_f, steps.to_f].min
          bounded_step * ((end_value - init) / steps.to_f) + init
        end
      end
    end

    class << self
      %i[exponential_decay step_decay cosine_decay join_schedules linear_schedule].each do |name|
        define_method(name) { |*args| Schedulers.public_send(name, *args) }
      end
    end
  end
end
