# frozen_string_literal: true

require "mlx"

class StreamingAffine < MLX::DSL::Model
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

model = StreamingAffine.new(in_dim: 1, out_dim: 1)
optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.05)

trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  pred = model.call(x)
  MLX::NN.mse_loss(pred, y, reduction: "mean")
end

train_data = lambda do |epoch, kind:|
  raise ArgumentError, "unexpected kind: #{kind.inspect}" unless kind == :train

  [batch(epoch + 1.0, 0.0), batch(epoch + 2.0, 0.0)]
end

report = trainer.fit_report(
  train_data,
  epochs: 3,
  strict_data_reuse: true,
  reduce: :mean
)

puts "epochs_ran=#{report.fetch('epochs_ran')} best_metric=#{report.fetch('best_metric')}"
