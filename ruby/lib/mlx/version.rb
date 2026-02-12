# frozen_string_literal: true

module MLX
  module Version
    module_function

    def from_header
      header = File.expand_path("../../../mlx/version.h", __dir__)
      return "0.0.0" unless File.file?(header)

      major = nil
      minor = nil
      patch = nil

      File.foreach(header) do |line|
        major = Regexp.last_match(1) if line =~ /#define\s+MLX_VERSION_MAJOR\s+(\d+)/
        minor = Regexp.last_match(1) if line =~ /#define\s+MLX_VERSION_MINOR\s+(\d+)/
        patch = Regexp.last_match(1) if line =~ /#define\s+MLX_VERSION_PATCH\s+(\d+)/
      end

      return "0.0.0" if [major, minor, patch].any?(&:nil?)

      "#{major}.#{minor}.#{patch}"
    end
  end

  VERSION = Version.from_header
end
