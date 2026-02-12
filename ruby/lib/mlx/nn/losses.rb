# frozen_string_literal: true

module MLX
  module NN
    module Losses
      module_function

      VALID_REDUCTIONS = %w[none mean sum].freeze

      def reduction(value, mode)
        mode_name = (mode || "none").to_s
        unless VALID_REDUCTIONS.include?(mode_name)
          raise ArgumentError, "Invalid reduction. Must be one of #{VALID_REDUCTIONS}."
        end

        return MLX::Core.mean(value) if mode_name == "mean"
        return MLX::Core.sum(value) if mode_name == "sum"

        value
      end

      def cross_entropy(
        logits,
        targets,
        weights: nil,
        axis: -1,
        label_smoothing: 0.0,
        reduction: "none"
      )
        if label_smoothing < 0 || label_smoothing >= 1
          raise ArgumentError, "Label smoothing must in [0, 1), got #{label_smoothing}."
        end

        targets_as_probs = targets.ndim == logits.ndim
        dropped_shape = drop_dim(logits.shape, axis)
        if (targets_as_probs && targets.shape != logits.shape) || (!targets_as_probs && targets.shape != dropped_shape)
          raise ArgumentError, "Targets shape #{targets.shape} does not match logits shape #{logits.shape}."
        end

        score = if targets_as_probs
          MLX::Core.sum(MLX::Core.multiply(logits, targets), axis)
        else
          idx = MLX::Core.expand_dims(targets, axis)
          MLX::Core.squeeze(MLX::Core.take_along_axis(logits, idx, axis), axis)
        end

        logsumexp_logits = MLX::Core.logsumexp(logits, axis)
        loss = if label_smoothing > 0
          adjusted_score = MLX::Core.multiply(score, 1 - label_smoothing)
          mean_logits = MLX::Core.mean(logits, axis)
          smoothed_loss = MLX::Core.multiply(mean_logits, -label_smoothing)
          MLX::Core.add(MLX::Core.subtract(logsumexp_logits, adjusted_score), smoothed_loss)
        else
          MLX::Core.subtract(logsumexp_logits, score)
        end

        unless weights.nil?
          if weights.shape != loss.shape
            raise ArgumentError,
                  "Weights with shape #{weights.shape} is not the same as output loss with shape #{loss.shape}."
          end
          loss = MLX::Core.multiply(loss, weights)
        end

        reduction(loss, reduction)
      end

      def binary_cross_entropy(
        inputs,
        targets,
        weights: nil,
        with_logits: true,
        reduction: "mean"
      )
        if inputs.shape != targets.shape
          raise ArgumentError, "Inputs shape #{inputs.shape} does not match targets shape #{targets.shape}."
        end

        loss = if with_logits
          MLX::Core.subtract(MLX::Core.logaddexp(0.0, inputs), MLX::Core.multiply(inputs, targets))
        else
          log_inputs_clip = MLX::Core.clip(MLX::Core.log(inputs), -100, nil)
          log_inputs_inv_clip = MLX::Core.clip(
            MLX::Core.log(MLX::Core.subtract(1.0, inputs)),
            -100,
            nil
          )
          weighted_terms = MLX::Core.add(
            MLX::Core.multiply(targets, log_inputs_clip),
            MLX::Core.multiply(MLX::Core.subtract(1.0, targets), log_inputs_inv_clip)
          )
          MLX::Core.multiply(weighted_terms, -1.0)
        end

        unless weights.nil?
          if weights.shape != loss.shape
            raise ArgumentError,
                  "Weights with shape #{weights.shape} is not the same as output loss with shape #{loss.shape}."
          end
          loss = MLX::Core.multiply(loss, weights)
        end

        reduction(loss, reduction)
      end

      def l1_loss(predictions, targets, reduction: "mean")
        if predictions.shape != targets.shape
          raise ArgumentError, "Predictions shape #{predictions.shape} does not match targets shape #{targets.shape}."
        end

        reduction(MLX::Core.abs(MLX::Core.subtract(predictions, targets)), reduction)
      end

      def mse_loss(predictions, targets, reduction: "mean")
        if predictions.shape != targets.shape
          raise ArgumentError, "Predictions shape #{predictions.shape} does not match targets shape #{targets.shape}."
        end

        diff = MLX::Core.subtract(predictions, targets)
        reduction(MLX::Core.square(diff), reduction)
      end

      def nll_loss(inputs, targets, axis: -1, reduction: "none")
        idx = MLX::Core.expand_dims(targets, -1)
        selected = MLX::Core.take_along_axis(inputs, idx, axis)
        loss = MLX::Core.multiply(MLX::Core.squeeze(selected, -1), -1.0)
        reduction(loss, reduction)
      end

      def gaussian_nll_loss(inputs, targets, vars, full: false, eps: 1e-6, reduction: "mean")
        if inputs.shape != targets.shape
          raise ArgumentError, "Inputs shape #{inputs.shape} does not match targets shape #{targets.shape}."
        end
        if inputs.shape != vars.shape
          raise ArgumentError, "Inputs shape #{inputs.shape} does not match vars shape #{vars.shape}."
        end

        vars = MLX::Core.maximum(vars, eps)
        squared_error = MLX::Core.square(MLX::Core.subtract(targets, inputs))
        base = MLX::Core.add(MLX::Core.log(vars), MLX::Core.divide(squared_error, vars))
        loss = MLX::Core.multiply(base, 0.5)
        if full
          loss = MLX::Core.add(loss, 0.5 * Math.log(2 * Math::PI))
        end

        reduction(loss, reduction)
      end

      def kl_div_loss(inputs, targets, axis: -1, reduction: "none")
        diff = MLX::Core.subtract(targets, inputs)
        loss = MLX::Core.sum(MLX::Core.multiply(MLX::Core.exp(targets), diff), axis)
        reduction(loss, reduction)
      end

      def smooth_l1_loss(predictions, targets, beta: 1.0, reduction: "mean")
        if predictions.shape != targets.shape
          raise ArgumentError, "Predictions shape #{predictions.shape} does not match targets shape #{targets.shape}."
        end

        diff = MLX::Core.abs(MLX::Core.subtract(predictions, targets))
        loss = MLX::Core.where(
          MLX::Core.less(diff, beta),
          MLX::Core.divide(MLX::Core.multiply(MLX::Core.square(diff), 0.5), beta),
          MLX::Core.subtract(MLX::Core.abs(diff), 0.5 * beta)
        )
        reduction(loss, reduction)
      end

      def triplet_loss(
        anchors,
        positives,
        negatives,
        axis: -1,
        p: 2,
        margin: 1.0,
        eps: 1e-6,
        reduction: "none"
      )
        ap = MLX::Core.subtract(anchors, positives)
        an = MLX::Core.subtract(anchors, negatives)
        dist_ap = MLX::Core.sqrt(MLX::Core.add(MLX::Core.sum(MLX::Core.power(ap, p), axis), eps))
        dist_an = MLX::Core.sqrt(MLX::Core.add(MLX::Core.sum(MLX::Core.power(an, p), axis), eps))
        loss = MLX::Core.maximum(MLX::Core.add(MLX::Core.subtract(dist_ap, dist_an), margin), 0.0)
        reduction(loss, reduction)
      end

      def hinge_loss(inputs, targets, reduction: "none")
        loss = MLX::Core.maximum(MLX::Core.subtract(1.0, MLX::Core.multiply(inputs, targets)), 0.0)
        reduction(loss, reduction)
      end

      def huber_loss(inputs, targets, delta: 1.0, reduction: "none")
        errors = MLX::Core.subtract(inputs, targets)
        abs_errors = MLX::Core.abs(errors)
        quadratic = MLX::Core.minimum(abs_errors, delta)
        linear = MLX::Core.subtract(abs_errors, quadratic)
        loss = MLX::Core.add(
          MLX::Core.multiply(MLX::Core.square(quadratic), 0.5),
          MLX::Core.multiply(linear, delta)
        )
        reduction(loss, reduction)
      end

      def log_cosh_loss(inputs, targets, reduction: "none")
        errors = MLX::Core.subtract(inputs, targets)
        loss = MLX::Core.subtract(MLX::Core.logaddexp(errors, MLX::Core.multiply(errors, -1.0)), Math.log(2))
        reduction(loss, reduction)
      end

      def cosine_similarity_loss(x1, x2, axis: 1, eps: 1e-8, reduction: "none")
        x1_norm = MLX::Core.sqrt(MLX::Core.sum(MLX::Core.square(x1), axis))
        x2_norm = MLX::Core.sqrt(MLX::Core.sum(MLX::Core.square(x2), axis))
        dot = MLX::Core.sum(MLX::Core.multiply(x1, x2), axis)
        denom = MLX::Core.maximum(MLX::Core.multiply(x1_norm, x2_norm), eps)
        loss = MLX::Core.divide(dot, denom)
        reduction(loss, reduction)
      end

      def margin_ranking_loss(inputs1, inputs2, targets, margin: 0.0, reduction: "none")
        unless inputs1.shape == inputs2.shape && inputs1.shape == targets.shape
          raise ArgumentError,
                "The shapes of the arguments do not match. The provided shapes are " \
                "inputs1.shape=#{inputs1.shape}, inputs2.shape=#{inputs2.shape}, and " \
                "targets.shape=#{targets.shape}."
        end

        differences = MLX::Core.subtract(inputs1, inputs2)
        term = MLX::Core.add(MLX::Core.multiply(MLX::Core.multiply(targets, differences), -1.0), margin)
        loss = MLX::Core.maximum(0.0, term)
        reduction(loss, reduction)
      end

      def drop_dim(shape, axis)
        axis_index = axis.negative? ? axis + shape.length : axis
        copied = shape.dup
        copied.delete_at(axis_index)
        copied
      end
    end

    class << self
      %i[
        cross_entropy binary_cross_entropy l1_loss mse_loss nll_loss gaussian_nll_loss
        kl_div_loss smooth_l1_loss triplet_loss hinge_loss huber_loss log_cosh_loss
        cosine_similarity_loss margin_ranking_loss
      ].each do |name|
        define_method(name) { |*args, **kwargs| Losses.public_send(name, *args, **kwargs) }
      end
    end
  end
end
