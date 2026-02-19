# frozen_string_literal: true

class GemTask
  REPO_ROOT = File.expand_path("..", __dir__).freeze
  VERSION_FILE = File.join(REPO_ROOT, "lib", "mlx", "version.rb").freeze

  def self.build!
    run_command!(["gem", "build", "mlx.gemspec"], chdir: REPO_ROOT)
  end

  def self.bump_version!
    content = File.read(VERSION_FILE)
    version_pattern = /^(\s*VERSION\s*=\s*")([^"]+)(")\s*$/
    match = content.match(version_pattern)

    raise "Could not find VERSION assignment in #{VERSION_FILE}" unless match

    old_version = match[2]
    segments = old_version.split(".")
    unless segments.all? { |segment| segment.match?(/\A\d+\z/) } && segments.length <= 4
      raise "Expected VERSION in numeric dotted format with up to 4 segments, got #{old_version.inspect}"
    end

    numeric_segments = segments.map(&:to_i)
    numeric_segments << 0 while numeric_segments.length < 4
    numeric_segments[3] += 1
    new_version = numeric_segments.join(".")

    updated = content.sub(version_pattern) { "#{Regexp.last_match(1)}#{new_version}#{Regexp.last_match(3)}" }
    File.write(VERSION_FILE, updated)

    puts "Bumped version: #{old_version} -> #{new_version}"
  end

  def self.run_command!(command, chdir:)
    success = system(*command, chdir: chdir)
    return if success

    raise "command failed: #{command.join(' ')} (cwd: #{chdir})"
  end
end
