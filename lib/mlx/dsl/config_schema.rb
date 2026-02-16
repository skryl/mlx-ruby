# frozen_string_literal: true

module MLX
  module DSL
    module ConfigSchema
      UNSET = Object.new.freeze
      class DefaultContext
        def initialize(values)
          @values = values
        end

        def method_missing(name, *args, &block)
          if args.empty? && block.nil? && @values.key?(name.to_sym)
            return @values[name.to_sym]
          end

          super
        end

        def respond_to_missing?(name, include_private = false)
          @values.key?(name.to_sym) || super
        end
      end

      def self.included(base)
        base.extend(ClassMethods)
      end

      module ClassMethods
        def field(name, type = nil, required: false, default: UNSET, &validator)
          key = name.to_sym
          config_schema_fields[key] = {
            type: type,
            required: !!required,
            default: default,
            validator: validator
          }
          attr_accessor key unless method_defined?(key) && method_defined?(:"#{key}=")
        end

        def config_schema_fields
          @config_schema_fields ||= {}
        end

        def inherited(subclass)
          super
          copied = config_schema_fields.each_with_object({}) do |(key, value), out|
            out[key] = value.dup
          end
          subclass.instance_variable_set(:@config_schema_fields, copied)
        end

        def from_hash(raw)
          source = (raw || {}).each_with_object({}) do |(key, value), out|
            out[key.to_sym] = value
          end
          new(**source)
        end

        private

        def __dsl_call_default(default, resolved)
          context = DefaultContext.new(resolved)
          return default unless default.respond_to?(:call)
          return default.call(context) if default.is_a?(Proc) && default.arity == 1
          return default.call if !default.is_a?(Proc)
          return default.call if default.arity.zero?

          default.call(context)
        end

        def __dsl_validate_field(name, value, spec)
          type = spec.fetch(:type)
          if !type.nil? && !value.nil?
            allowed_types = type.is_a?(Array) ? type : [type]
            unless allowed_types.any? { |klass| value.is_a?(klass) }
              raise TypeError,
                    "config field #{name} must be #{allowed_types.map(&:to_s).join(' or ')}, got #{value.class}"
            end
          end

          validator = spec.fetch(:validator)
          unless validator.nil?
            if validator.arity == 2
              validator.call(value, name)
            else
              validator.call(value)
            end
          end

          value
        end
      end

      def initialize(**kwargs)
        source = kwargs.each_with_object({}) do |(key, value), out|
          out[key.to_sym] = value
        end
        resolved = {}
        unknown = source.keys - self.class.config_schema_fields.keys
        unless unknown.empty?
          names = unknown.map(&:to_s).sort.join(", ")
          raise ArgumentError, "unknown config field(s): #{names}"
        end

        self.class.config_schema_fields.each do |name, spec|
          if source.key?(name)
            value = source.fetch(name)
          else
            default = spec.fetch(:default)
            if default.equal?(UNSET)
              if spec.fetch(:required)
                raise ArgumentError, "missing required config field: #{name}"
              end
              next
            end
            value = self.class.send(:__dsl_call_default, default, resolved)
          end

          value = self.class.send(:__dsl_validate_field, name, value, spec)
          resolved[name] = value
          public_send(:"#{name}=", value)
        end
      end

      def to_h
        self.class.config_schema_fields.keys.each_with_object({}) do |name, out|
          out[name.to_s] = public_send(name)
        end
      end
    end
  end
end
