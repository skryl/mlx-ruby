# frozen_string_literal: true

require "etc"
require "fileutils"
require "mkmf"

def run_or_abort(*cmd, chdir:)
  puts ">> #{cmd.join(' ')}"
  success = system(*cmd, chdir: chdir)
  return if success

  abort("command failed in #{chdir}: #{cmd.join(' ')}")
end

def run_with_status(*cmd, chdir:)
  puts ">> #{cmd.join(' ')}"
  system(*cmd, chdir: chdir)
end

def rpath_flag(path)
  case RUBY_PLATFORM
  when /darwin/
    "-Wl,-rpath,#{path}"
  when /linux/
    "-Wl,-rpath,#{path}"
  else
    ""
  end
end

repo_root = File.expand_path("../..", __dir__)
mlx_root = File.join(repo_root, "mlx")
mlx_include_dir = mlx_root
ext_root = File.expand_path(__dir__)
build_root = File.join(ext_root, "build")
mlx_build_dir = File.join(build_root, "mlx")
mlx_install_dir = File.join(build_root, "install")
jobs = [Etc.nprocessors, 1].max

FileUtils.mkdir_p(mlx_build_dir)

cmake_configure = [
  "cmake",
  "-S",
  mlx_root,
  "-B",
  mlx_build_dir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=#{mlx_install_dir}",
  "-DMLX_BUILD_TESTS=OFF",
  "-DMLX_BUILD_EXAMPLES=OFF",
  "-DMLX_BUILD_BENCHMARKS=OFF",
  "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
  "-DMLX_BUILD_PYTHON_STUBS=OFF",
  "-DMLX_BUILD_METAL=ON",
  "-DMLX_BUILD_GGUF=OFF",
  "-DMLX_BUILD_SAFETENSORS=OFF",
  "-DBUILD_SHARED_LIBS=ON"
]

cmake_build = [
  "cmake",
  "--build",
  mlx_build_dir,
  "--target",
  "install",
  "--config",
  "Release",
  "-j#{jobs}"
]

configured = run_with_status(*cmake_configure, chdir: ext_root)
unless configured
  warn "initial CMake configure failed; cleaning build tree and retrying once"
  FileUtils.rm_rf(build_root)
  FileUtils.mkdir_p(mlx_build_dir)
  run_or_abort(*cmake_configure, chdir: ext_root)
end
run_or_abort(*cmake_build, chdir: ext_root)

include_dir = mlx_include_dir
lib_dir = File.join(mlx_install_dir, "lib")

abort("missing MLX include dir: #{include_dir}") unless Dir.exist?(include_dir)
abort("missing MLX lib dir: #{lib_dir}") unless Dir.exist?(lib_dir)

dir_config("mlx", include_dir, lib_dir)

$CXXFLAGS = "#{$CXXFLAGS} -std=c++20"
$CPPFLAGS = "#{$CPPFLAGS} -I#{include_dir}"
$LDFLAGS = "#{$LDFLAGS} -L#{lib_dir} #{rpath_flag(lib_dir)}"
$libs = "-lmlx #{$libs}"

create_makefile("mlx/native")
