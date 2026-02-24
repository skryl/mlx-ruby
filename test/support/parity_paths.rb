# frozen_string_literal: true

module TestSupport
  module ParityPaths
    def parity_generated_reports_dir
      @parity_generated_reports_dir ||= begin
        path = File.join(RUBY_ROOT, "test", "reports")
        FileUtils.mkdir_p(path)
        path
      end
    end

    def parity_generated_report_path(name)
      File.join(parity_generated_reports_dir, name)
    end

    def parity_snapshot_dir
      File.join(RUBY_ROOT, "test", "support", "snapshots", "parity")
    end

    def parity_snapshot_path(name)
      File.join(parity_snapshot_dir, name)
    end
  end
end
