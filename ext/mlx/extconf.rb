# frozen_string_literal: true

require "etc"
require "fileutils"
require "mkmf"
require "open3"
require "pathname"
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

def git_revision(path)
  return nil unless Dir.exist?(path)

  stdout, status = Open3.capture2("git", "-C", path, "rev-parse", "HEAD")
  return nil unless status.success?

  rev = stdout.strip
  return rev if rev.match?(/\A[0-9a-f]{40}\z/)

  nil
end

def mlx_revision_pinned_by_mlx_onnx(mlx_onnx_root)
  return nil unless Dir.exist?(mlx_onnx_root)

  stdout, status = Open3.capture2("git", "-C", mlx_onnx_root, "submodule", "status", "--", "mlx")
  return nil unless status.success?

  line = stdout.lines.first.to_s.strip
  return nil if line.empty?

  token = line.split(/\s+/).first.to_s
  token = token.delete_prefix("-").delete_prefix("+").delete_prefix("U")
  return token if token.match?(/\A[0-9a-f]{40}\z/)

  nil
end

def enforce_mlx_onnx_compatibility!(mlx_root:, mlx_onnx_root:)
  workspace_mlx_revision =
    ENV.fetch("MLX_EXTCONF_TEST_MLX_REVISION", "").strip
  pinned_mlx_revision =
    ENV.fetch("MLX_EXTCONF_TEST_MLX_ONNX_PINNED_MLX_REVISION", "").strip

  workspace_mlx_revision = git_revision(mlx_root) if workspace_mlx_revision.empty?
  pinned_mlx_revision = mlx_revision_pinned_by_mlx_onnx(mlx_onnx_root) if pinned_mlx_revision.empty?

  return if workspace_mlx_revision.nil? || pinned_mlx_revision.nil?
  return if workspace_mlx_revision == pinned_mlx_revision

  abort(<<~MSG)
    mlx/mlx-onnx revision mismatch detected.
    workspace mlx revision: #{workspace_mlx_revision}
    mlx-onnx pinned mlx revision: #{pinned_mlx_revision}
    Run:
      git submodule update --init --recursive submodules/mlx submodules/mlx-onnx
  MSG
end

def patch_mlx_onnx_gcc_optional_shape_initlist!(mlx_onnx_root)
  lowering_cpp = File.join(mlx_onnx_root, "src", "lowering.cpp")
  return unless File.file?(lowering_cpp)

  source = File.read(lowering_cpp)
  patched = source
    .gsub(
      "std::optional<Shape>({work_shape[0], 1, seq_len})",
      "std::optional<Shape>(Shape{work_shape[0], 1, seq_len})"
    )
    .gsub(
      "std::optional<Shape>({seq_len})",
      "std::optional<Shape>(Shape{seq_len})"
    )

  return if patched == source

  File.write(lowering_cpp, patched)
  puts "patched mlx-onnx lowering.cpp optional Shape initlists for GCC compatibility"
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

def patch_makefile_sources!(makefile_path, source_files, header_files)
  return unless File.file?(makefile_path)

  objects = source_files.map do |path|
    stem = path.sub(/\.[^.]+\z/, "")
    "#{stem}.o"
  end
  text = File.read(makefile_path)
  text = text.gsub(/^ORIG_SRCS = .+$/, "ORIG_SRCS = #{source_files.join(' ')}")
  text = text.gsub(/^OBJS = .+$/, "OBJS = #{objects.join(' ')}")
  text = text.gsub(/^HDRS = .+$/, "HDRS = #{header_files.map { |path| '$(srcdir)/' + path }.join(' ')}")
  File.write(makefile_path, text)
end

def patch_makefile_include_dirs!(makefile_path, include_dirs)
  return unless File.file?(makefile_path)

  text = File.read(makefile_path)
  match = text.match(/^CPPFLAGS = (.+)$/)
  return if match.nil?

  cppflags = match[1]
  include_dirs.each do |path|
    flag = "-I#{path}"
    cppflags = "#{cppflags} #{flag}" unless cppflags.include?(flag)
  end
  text = text.sub(/^CPPFLAGS = .+$/, "CPPFLAGS = #{cppflags}")
  File.write(makefile_path, text)
end

repo_root = File.expand_path("../..", __dir__)
submodules_root = File.join(repo_root, "submodules")
mlx_root = File.join(submodules_root, "mlx")
mlx_onnx_root = File.join(submodules_root, "mlx-onnx")
mlx_public_include_dir = mlx_root
mlx_internal_include_dir = File.join(mlx_root, "mlx")
mlx_onnx_public_include_dir = File.join(mlx_onnx_root, "include")
mlx_onnx_internal_include_dir = File.join(mlx_onnx_root, "src")
ext_root = File.expand_path(__dir__)
ext_onnx_root = File.join(repo_root, "ext", "mlx-onnx")
build_root = File.join(ext_root, "build")
mlx_build_dir = File.join(build_root, "mlx")
mlx_onnx_build_dir = File.join(build_root, "mlx-onnx")
mlx_install_dir = File.join(build_root, "install")
jobs = [Etc.nprocessors, 1].max

enforce_mlx_onnx_compatibility!(mlx_root: mlx_root, mlx_onnx_root: mlx_onnx_root)
patch_mlx_onnx_gcc_optional_shape_initlist!(mlx_onnx_root)
if ENV["MLX_EXTCONF_VALIDATE_ONLY"] == "1"
  puts "mlx-onnx compatibility check passed"
  exit 0
end

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
  "-DMLX_BUILD_SAFETENSORS=ON",
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

abort("missing MLX include dir: #{mlx_public_include_dir}") unless Dir.exist?(mlx_public_include_dir)
abort("missing MLX internal include dir: #{mlx_internal_include_dir}") unless Dir.exist?(mlx_internal_include_dir)
abort("missing mlx-onnx include dir: #{mlx_onnx_public_include_dir}") unless Dir.exist?(mlx_onnx_public_include_dir)
abort("missing mlx-onnx internal include dir: #{mlx_onnx_internal_include_dir}") unless Dir.exist?(mlx_onnx_internal_include_dir)
include_dirs = [
  mlx_public_include_dir,
  mlx_internal_include_dir,
  mlx_onnx_public_include_dir,
  mlx_onnx_internal_include_dir
]
lib_dir = File.join(mlx_install_dir, "lib")

cmake_onnx_configure = [
  "cmake",
  "-S",
  mlx_onnx_root,
  "-B",
  mlx_onnx_build_dir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=#{mlx_install_dir}",
  "-DMLX_ONNX_USE_EXTERNAL_MLX=ON",
  "-DMLX_ONNX_EXTERNAL_MLX_INCLUDE_DIR=#{mlx_public_include_dir}",
  "-DMLX_ONNX_EXTERNAL_MLX_LIB_DIR=#{lib_dir}",
  "-DMLX_ONNX_BUILD_PYTHON_BINDINGS=OFF"
]

cmake_onnx_build = [
  "cmake",
  "--build",
  mlx_onnx_build_dir,
  "--target",
  "install",
  "--config",
  "Release",
  "-j#{jobs}"
]

run_or_abort(*cmake_onnx_configure, chdir: ext_root)
run_or_abort(*cmake_onnx_build, chdir: ext_root)

json_include_dirs = [
  File.join(mlx_build_dir, "_deps", "json-src", "include"),
  File.join(mlx_build_dir, "_deps", "json-src", "single_include"),
  File.join(mlx_build_dir, "_deps", "json-src", "single_include", "nlohmann"),
  File.join(mlx_onnx_build_dir, "_deps", "nlohmann_json-src", "include")
].select { |path| Dir.exist?(path) }

abort("missing MLX lib dir: #{lib_dir}") unless Dir.exist?(lib_dir)

dir_config("mlx", include_dirs.first, lib_dir)

$CXXFLAGS = "#{$CXXFLAGS} -std=c++20"
include_dirs.each do |path|
  $CPPFLAGS = "#{$CPPFLAGS} -I#{path}"
end
json_include_dirs.each do |path|
  $CPPFLAGS = "#{$CPPFLAGS} -I#{path}"
end
$LDFLAGS = "#{$LDFLAGS} -L#{lib_dir} #{rpath_flag(lib_dir)}"
$libs = "-lmlx_onnx -lmlx #{$libs}"

source_files = (
  Dir.glob(File.join(ext_root, "**", "*.{c,cc,cpp,cxx}")) +
  Dir.glob(File.join(ext_onnx_root, "**", "*.{c,cc,cpp,cxx}"))
).reject { |path| path.start_with?(File.join(build_root, "")) }
 .map { |path| Pathname(path).relative_path_from(Pathname(ext_root)).to_s }
 .sort
header_files = (
  Dir.glob(File.join(ext_root, "**", "*.{h,hpp,hh}")) +
  Dir.glob(File.join(ext_onnx_root, "**", "*.{h,hpp,hh}"))
).reject { |path| path.start_with?(File.join(build_root, "")) }
 .map { |path| Pathname(path).relative_path_from(Pathname(ext_root)).to_s }
 .sort
$srcs = source_files
$objs = source_files.map { |path| "#{path.sub(/\.[^.]+\z/, "")}.o" }
$hdrs = header_files

create_makefile("mlx/native")
makefile_path = File.join(ext_root, "Makefile")
patch_makefile_compilers!(makefile_path, cmake_cc, cmake_cxx)
patch_makefile_sources!(makefile_path, source_files, header_files)
patch_makefile_include_dirs!(
  makefile_path,
  [
    mlx_internal_include_dir,
    mlx_onnx_public_include_dir,
    mlx_onnx_internal_include_dir,
    File.join(mlx_build_dir, "_deps", "json-src", "single_include", "nlohmann"),
    File.join(mlx_onnx_build_dir, "_deps", "nlohmann_json-src", "include")
  ]
)
