# frozen_string_literal: true

module TestSupport
  module TmpHelpers
    def test_tmp_dir
      @test_tmp_dir ||= begin
        path = File.join(RUBY_ROOT, "test", "tmp")
        FileUtils.mkdir_p(path)
        path
      end
    end

    def mktmpdir(prefix = "mlx-ruby-")
      return Dir.mktmpdir(prefix, test_tmp_dir) unless block_given?

      Dir.mktmpdir(prefix, test_tmp_dir) do |dir|
        yield dir
      end
    end
  end
end
