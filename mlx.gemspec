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

  spec.required_ruby_version = ">= 3.3"

  spec.files = Dir.chdir(__dir__) do
    include_globs = [
      "lib/*.rb",
      "lib/mlx/**/*",
      "lib/mlx-onnx/**/*",
      "ext/mlx/**/*.{rb,c,cc,cpp,cxx,h,hpp,hh}",
      "ext/mlx-onnx/**/*.{rb,c,cc,cpp,cxx,h,hpp,hh}",
      "submodules/mlx/CMakeLists.txt",
      "submodules/mlx/mlx.pc.in",
      "submodules/mlx/cmake/**/*",
      "submodules/mlx/mlx/**/*"
    ]

    Dir.glob(include_globs, File::FNM_DOTMATCH)
      .reject { |path| File.directory?(path) }
      .reject { |path| path.start_with?("ext/mlx/build/") }
      .reject { |path| path.end_with?(".o", ".bundle") }
      .uniq
      .sort
  end
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/mlx/extconf.rb"]
  spec.add_development_dependency "rake"
  spec.add_development_dependency "base64"
  spec.add_development_dependency "ostruct"
  spec.add_development_dependency "minitest"
  spec.add_development_dependency "benchmark"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["github_uri"] = "https://github.com/skryl"
end
