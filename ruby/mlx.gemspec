# frozen_string_literal: true

require_relative "lib/mlx/version"

Gem::Specification.new do |spec|
  spec.name = "mlx-ruby"
  spec.version = MLX::VERSION
  spec.authors = ["MLX Contributors"]
  spec.email = ["mlx@group.apple.com"]

  spec.summary = "Ruby bindings for the native MLX library"
  spec.description = "A Ruby wrapper for the native MLX machine learning runtime."
  spec.homepage = "https://github.com/ml-explore/mlx"
  spec.license = "MIT"

  spec.required_ruby_version = ">= 3.1"

  spec.files = Dir.glob("{lib,ext,tools,parity,test}/**/*", File::FNM_DOTMATCH)
    .reject { |path| File.directory?(path) }
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/mlx/extconf.rb"]

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
end
