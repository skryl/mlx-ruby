# frozen_string_literal: true

module MLX
  module DSL
  end
end

require_relative "dsl/graph_modules"
require_relative "dsl/builder"
require_relative "dsl/train_step"
require_relative "dsl/model_mixin"
require_relative "dsl/model"
require_relative "dsl/trainer"
