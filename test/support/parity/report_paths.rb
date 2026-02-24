#!/usr/bin/env ruby
# frozen_string_literal: true

require "pathname"

module ParityReportPaths
  module_function

  def repo_root
    Pathname.new(File.expand_path("../../..", __dir__))
  end

  def generated_root
    env_path = ENV["MLX_PARITY_REPORT_ROOT"]
    root = if env_path.nil? || env_path.strip.empty?
      repo_root.join("test", "reports")
    else
      Pathname.new(File.expand_path(env_path))
    end
    root.mkpath
    root
  end

  def snapshot_root
    repo_root.join("test", "support", "snapshots", "parity")
  end
end
