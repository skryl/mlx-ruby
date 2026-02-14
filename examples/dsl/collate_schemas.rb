# frozen_string_literal: true

require "mlx"

class CollateAffine < MLX::DSL::Model
  option :in_dim
  option :out_dim

  param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape, _dtype) { MLX::Core.ones(shape, MLX::Core.float32) }
  buffer :offset, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.zeros(shape, MLX::Core.float32) }

  def call(x)
    MLX::Core.add(MLX::Core.matmul(x, weight.T), offset)
  end
end

def scalar_batch(x_value, y_value)
  [MLX::Core.array([[x_value]], MLX::Core.float32), MLX::Core.array([[y_value]], MLX::Core.float32)]
end

model = CollateAffine.new(in_dim: 1, out_dim: 1)
optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.02)

trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  pred = model.call(x)
  MLX::NN.mse_loss(pred, y, reduction: "mean")
end

xy_report = trainer.fit_report(
  [scalar_batch(1.0, 0.0), scalar_batch(2.0, 0.0)],
  epochs: 1,
  collate: :xy
)

mapped_report = trainer.fit_report(
  [[MLX::Core.array([[3.0]], MLX::Core.float32), MLX::Core.array([[0.0]], MLX::Core.float32)]],
  epochs: 1,
  collate: { x: 0, y: 1 }
)

puts "xy_epoch_loss=#{xy_report.fetch('epochs')[0].fetch('epoch_loss')}"
puts "mapped_epoch_loss=#{mapped_report.fetch('epochs')[0].fetch('epoch_loss')}"
