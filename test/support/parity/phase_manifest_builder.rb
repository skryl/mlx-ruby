# frozen_string_literal: true

require "pathname"
require "time"

module PhaseManifestBuilder
  module_function

  DOMAIN_RULES = [
    ["perf", /(?:_perf_test\.rb\z|benchmark|timing|perf)/],
    ["distributed", /(?:distributed|dlpack|launch|mpi|remote|config)/],
    ["optimizers", /(?:optimizer|adam|adamw|adamax|sgd|rmsprop|adagrad|adadelta|lion|muon|adafactor|clip_grad_norm|scheduler)/],
    ["nn", /(?:module|loss|initializer|linear|embedding|dropout|activation|positional|convolution|pooling|normalization|recurrent|transformer|upsample|nn_)/],
    ["core", /.*/]
  ].freeze

  def build(repo_root:)
    root = File.expand_path(repo_root)
    phase_files = Dir.glob(File.join(root, "test", "parity", "phase*_test.rb")).sort

    phases = {}
    phase_files.each do |path|
      phase_id = extract_phase_id(path)
      next if phase_id.nil?

      phases[phase_id] = {
        "file" => relative_path(path, root),
        "domain" => infer_domain(path),
        "methods" => parse_test_methods(path)
      }
    end

    {
      "format" => "mlx_parity_phase_manifest_v1",
      "generated_at" => Time.now.utc.iso8601,
      "phases" => phases.sort.to_h
    }
  end

  def extract_phase_id(path)
    File.basename(path)[/\Aphase(\d+)_/, 1]
  end

  def parse_test_methods(path)
    methods = []
    File.foreach(path) do |line|
      match = line.match(/^\s*def\s+(test_[A-Za-z0-9_!?]+)/)
      methods << match[1] if match
    end
    methods.uniq.sort
  end

  def infer_domain(path)
    basename = File.basename(path)
    DOMAIN_RULES.each do |domain, pattern|
      return domain if basename.match?(pattern)
    end
    "core"
  end

  def relative_path(path, repo_root)
    Pathname.new(path).relative_path_from(Pathname.new(repo_root)).to_s
  end
end
