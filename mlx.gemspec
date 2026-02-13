# frozen_string_literal: true

require_relative "lib/mlx/version.rb"

Gem::Specification.new do |spec|
  spec.name = "mlx"
  spec.version = MLX::VERSION
  spec.authors = ["MLX Contributors", "Aleksey Skryl"]
  spec.email = ["mlx@group.apple.com", "aleksey.skryl@gmail.com"]

  spec.summary = "Ruby bindings for the native MLX library"
  spec.description = "A Ruby wrapper for the native MLX machine learning runtime."
  spec.homepage = "https://github.com/skryl/mlx-ruby"
  spec.license = "MIT"

  spec.required_ruby_version = ">= 3.1"

  spec.files = Dir.chdir(__dir__) do
    include_globs = [
      "lib/**/*",
      "ext/mlx/extconf.rb",
      "ext/mlx/native.cpp",
      "mlx/CMakeLists.txt",
      "mlx/mlx.pc.in",
      "mlx/cmake/**/*",
      "mlx/mlx/**/*"
    ]

    Dir.glob(include_globs, File::FNM_DOTMATCH)
      .reject { |path| File.directory?(path) }
      .uniq
      .sort
  end
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/mlx/extconf.rb"]

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["github_uri"] = "https://github.com/skryl"
end
