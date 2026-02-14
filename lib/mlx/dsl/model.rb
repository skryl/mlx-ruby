# frozen_string_literal: true

module MLX
  module DSL
    class Model < MLX::NN::Module
      include ModelMixin
    end
  end
end
