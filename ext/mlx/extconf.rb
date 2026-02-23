# frozen_string_literal: true

require "etc"
require "fileutils"
require "mkmf"
require "rbconfig"

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

def cmake_compilers_from_cache(cache_path)
  return [nil, nil] unless File.file?(cache_path)

  cc = nil
  cxx = nil
  File.foreach(cache_path) do |line|
    cc = Regexp.last_match(1).strip if line =~ /^CMAKE_C_COMPILER:(?:STRING|FILEPATH)=(.+)$/
    cxx = Regexp.last_match(1).strip if line =~ /^CMAKE_CXX_COMPILER:(?:STRING|FILEPATH)=(.+)$/
  end
  [cc, cxx]
end

def normalize_compiler_for_mkmf(path, kind)
  return path if path.nil? || path.empty?

  case kind
  when :cc
    return "/usr/bin/clang" if path.end_with?("/usr/bin/cc")
  when :cxx
    return "/usr/bin/clang++" if path.end_with?("/usr/bin/c++")
  end
  path
end

def force_mkmf_compilers!(cc, cxx)
  return if cc.nil? || cc.empty? || cxx.nil? || cxx.empty?

  cc = normalize_compiler_for_mkmf(cc, :cc)
  cxx = normalize_compiler_for_mkmf(cxx, :cxx)

  [RbConfig::CONFIG, RbConfig::MAKEFILE_CONFIG].each do |cfg|
    cfg["CC"] = cc if cfg.key?("CC")
    cfg["CXX"] = cxx if cfg.key?("CXX")

    if cfg.key?("LDSHARED") && cfg["LDSHARED"]
      cfg["LDSHARED"] = cfg["LDSHARED"].sub(/\A\S+/, cc)
    end
    if cfg.key?("LDSHAREDXX") && cfg["LDSHAREDXX"]
      cfg["LDSHAREDXX"] = cfg["LDSHAREDXX"].sub(/\A\S+/, cxx)
    end
  end

  $CC = cc if defined?($CC)
  $CXX = cxx if defined?($CXX)
  $LDSHARED = $LDSHARED.sub(/\A\S+/, cc) if defined?($LDSHARED) && $LDSHARED
  $LDSHAREDXX = $LDSHAREDXX.sub(/\A\S+/, cxx) if defined?($LDSHAREDXX) && $LDSHAREDXX
end

def patch_makefile_compilers!(makefile_path, cc, cxx)
  return if cc.nil? || cc.empty? || cxx.nil? || cxx.empty?
  return unless File.file?(makefile_path)

  cc = normalize_compiler_for_mkmf(cc, :cc)
  cxx = normalize_compiler_for_mkmf(cxx, :cxx)

  text = File.read(makefile_path)
  text = text.gsub(/^CC = .+$/, "CC = #{cc}")
  text = text.gsub(/^CXX = .+$/, "CXX = #{cxx}")
  File.write(makefile_path, text)
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

cmake_cache_path = File.join(mlx_build_dir, "CMakeCache.txt")
cmake_cc, cmake_cxx = cmake_compilers_from_cache(cmake_cache_path)
force_mkmf_compilers!(cmake_cc, cmake_cxx)

include_dir = mlx_include_dir
lib_dir = File.join(mlx_install_dir, "lib")
json_include_dirs = [
  File.join(mlx_build_dir, "_deps", "json-src", "include"),
  File.join(mlx_build_dir, "_deps", "json-src", "single_include")
].select { |path| Dir.exist?(path) }

abort("missing MLX include dir: #{include_dir}") unless Dir.exist?(include_dir)
abort("missing MLX lib dir: #{lib_dir}") unless Dir.exist?(lib_dir)

dir_config("mlx", include_dir, lib_dir)

$CXXFLAGS = "#{$CXXFLAGS} -std=c++20"
$CPPFLAGS = "#{$CPPFLAGS} -I#{include_dir}"
json_include_dirs.each do |path|
  $CPPFLAGS = "#{$CPPFLAGS} -I#{path}"
end
$LDFLAGS = "#{$LDFLAGS} -L#{lib_dir} #{rpath_flag(lib_dir)}"
$libs = "-lmlx #{$libs}"

create_makefile("mlx/native")
patch_makefile_compilers!(File.join(Dir.pwd, "Makefile"), cmake_cc, cmake_cxx)
