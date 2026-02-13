# frozen_string_literal: true

require "mlx"

class ValidationAffine < MLX::DSL::Model
  option :in_dim
  option :out_dim

  param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape, _dtype) { MLX::Core.ones(shape, MLX::Core.float32) }
  buffer :offset, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.zeros(shape, MLX::Core.float32) }

  def call(x)
    MLX::Core.add(MLX::Core.matmul(x, weight.T), offset)
  end
end

def batch(x_value, y_value)
  {
    x: MLX::Core.array([[x_value]], MLX::Core.float32),
    y: MLX::Core.array([[y_value]], MLX::Core.float32)
  }
end

model = ValidationAffine.new(in_dim: 1, out_dim: 1)
optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.01)

trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  pred = model.call(x)
  MLX::NN.mse_loss(pred, y, reduction: "mean")
end

trainer.after_validation { |ctx| puts "epoch=#{ctx[:epoch]} val_loss=#{ctx[:val_loss]}" }

report = trainer.fit_report(
  [batch(1.0, 0.0), batch(2.0, 0.0)],
  epochs: 2,
  validation_data: [[1.0, 0.0], [3.0, 0.0]],
  validation_transform: ->(raw, epoch:) { batch(raw[0] + epoch, raw[1]) },
  validation_reduce: :mean,
  monitor: :val_loss,
  monitor_mode: :min,
  save_best: true
)

puts "monitor=#{report.fetch('monitor_name')} best=#{report.fetch('best_metric')}"
