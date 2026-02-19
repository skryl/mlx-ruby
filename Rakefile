# frozen_string_literal: true

require "rake"
require "rake/clean"
require "rake/testtask"

require_relative "tasks/benchmark_task"
require_relative "tasks/build_task"
require_relative "tasks/docs_task"
require_relative "tasks/gem_task"
require_relative "tasks/test_task"
require_relative "tasks/training_task"
require_relative "tasks/web_assets_task"
require_relative "tasks/web_task"

CLEAN.include(*BuildTask.clean_patterns)

if MlxTestTask.strict_mode?
  desc "Run tests with strict per-file timeout (set MLX_TEST_TIMEOUT to customize)."
  task :test_base do
    MlxTestTask.run_strict_test_suite!
  end
else
  Rake::TestTask.new(:test_base) do |t|
    MlxTestTask.configure_base_test_task(t)
  end
end

desc "Run test suite on cpu+gpu by default. Override devices: rake \"test[cpu]\" or rake \"test[gpu]\"."
task :test, [:devices] do |_task, args|
  MlxTestTask.run_test_suite_for_devices(args[:devices])
end

namespace :test do
  desc "Run the full test suite with DEVICE=cpu."
  task :cpu do
    MlxTestTask.run_test_suite_for_device(:cpu)
  end

  desc "Run the full test suite with DEVICE=gpu."
  task :gpu do
    MlxTestTask.run_test_suite_for_device(:gpu)
  end

  desc "Build, install, and run tests against the installed gem artifact."
  task :gem do
    MlxTestTask.run_installed_gem_test_suite!
  end
end

desc "Build native extension."
task :build do
  BuildTask.build_native_extension!
end

namespace :docs do
  desc "Build documentation (Doxygen + Sphinx HTML)."
  task :build do
    DocsTask.build!
  end
end

namespace :gem do
  desc "Build gem package from mlx.gemspec."
  task :build do
    GemTask.build!
  end

  desc "Bump gem version by 0.0.0.1 in lib/mlx/version.rb."
  task :bump do
    GemTask.bump_version!
  end
end

namespace :benchmark do
  desc "Install Python benchmark dependencies into the active Python from requirements.txt."
  task :deps do
    BenchmarkTask.install_dependencies!
  end

  desc "Run selected models on cpu. Usage: rake 'benchmark:cpu[local,examples]' or rake 'benchmark:cpu[model_a,model_b]'."
  task :cpu, [:models] do |_task, args|
    BenchmarkTask.run_cpu_task(args)
  end

  desc "Run selected models on gpu. Usage: rake 'benchmark:gpu[local,examples]' or rake 'benchmark:gpu[model_a,model_b]'."
  task :gpu, [:models] do |_task, args|
    BenchmarkTask.run_gpu_task(args)
  end

  desc "Run selected models on WebGPU/local GPU path. Usage: rake 'benchmark:webgpu[local,examples]' or rake 'benchmark:webgpu[model_a,model_b]'."
  task :webgpu, [:models] do |_task, args|
    BenchmarkTask.run_webgpu_task(args)
  end

  desc "Generate GraphIR WebGPU compatibility coverage artifact for benchmark fixtures."
  task :graph_ir_coverage do
    BenchmarkTask.run_graph_ir_coverage!
  end

  desc "Run selected models on cpu, gpu, and webgpu. Usage: rake 'benchmark:all[local,examples]' or rake 'benchmark:all[model_a,model_b]'."
  task :all, [:models] do |_task, args|
    BenchmarkTask.run_all_task(args)
  end
end

desc "Run selected benchmarks on cpu, gpu, and webgpu. Usage: rake 'benchmark[local,examples]' or rake 'benchmark[model_a,model_b]'."
task :benchmark, [:models] do |_task, args|
  BenchmarkTask.run_top_level_task(args)
end

namespace :web do
  desc "Export browser demo assets under web/assets."
  task :assets do
    WebAssetsTask.run!
  end

  desc "Train web demo model weights. Usage: rake 'web:train[model]' (models: nanogpt). Default model: nanogpt."
  task :train, [:model] do |_task, args|
    TrainingTask.run!(args[:model])
  end

  desc "Start local web server for the browser demos (http://127.0.0.1:3030/)."
  task :start do
    WebTask.start!
  end
end

task default: :test
