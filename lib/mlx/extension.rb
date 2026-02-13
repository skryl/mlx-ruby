# frozen_string_literal: true

module MLX
  module Extension
    class CMakeExtension
      attr_reader :name, :sourcedir

      def initialize(name, sourcedir = "")
        @name = name
        @sourcedir = sourcedir
      end
    end

    class CMakeBuild
      def build_extension(_ext)
        true
      end

      def run
        true
      end
    end
  end
end
