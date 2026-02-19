# frozen_string_literal: true

require_relative "web_assets_task"

class WebTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze
  WEB_ROOT = File.join(REPO_ROOT, "web").freeze

  def self.start!(
    host: ENV.fetch("HOST", "127.0.0.1"),
    port: Integer(ENV.fetch("PORT", "3030")),
    python_bin: ENV.fetch("PYTHON_BIN", ENV.fetch("PYTHON", "python3"))
  )
    missing_assets = required_assets.reject { |path| File.exist?(path) }
    unless missing_assets.empty?
      puts "Demo assets are missing; exporting now..."
      WebAssetsTask.run!
    end

    entry = "http://#{host}:#{port}/"
    puts "Serving #{WEB_ROOT}"
    puts "Demo index: #{entry}"
    puts "Press Ctrl+C to stop."
    exec python_bin, "-m", "http.server", port.to_s, "--bind", host, "--directory", WEB_ROOT
  end

  def self.required_assets
    [
      File.join(WEB_ROOT, "assets", "cvae", "model.onnx"),
      File.join(WEB_ROOT, "assets", "gpt2", "model.onnx"),
      File.join(WEB_ROOT, "assets", "nanogpt", "model.onnx")
    ]
  end
end
