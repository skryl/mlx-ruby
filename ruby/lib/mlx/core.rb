# frozen_string_literal: true

require "open3"
require "tmpdir"

module MLX
  module Core
    class NativeUnavailableError < StandardError; end

    module DeviceType
      module_function

      def cpu
        :cpu
      end

      def gpu
        :gpu
      end
    end

    class Finfo
      FLOAT_INFO = {
        "float16" => { min: -65_504.0, max: 65_504.0, eps: 9.765625e-4 },
        "bfloat16" => { min: -3.389531389e38, max: 3.389531389e38, eps: 7.8125e-3 },
        "float32" => { min: -3.4028235e38, max: 3.4028235e38, eps: 1.1920929e-7 },
        "float64" => { min: -Float::MAX, max: Float::MAX, eps: Float::EPSILON },
        "complex64" => { min: -3.4028235e38, max: 3.4028235e38, eps: 1.1920929e-7 }
      }.freeze

      attr_reader :dtype, :min, :max, :eps

      def initialize(dtype)
        @dtype = dtype
        info = FLOAT_INFO[dtype_name(dtype)]
        raise ArgumentError, "unsupported dtype for finfo: #{dtype_name(dtype)}" if info.nil?

        @min = info[:min]
        @max = info[:max]
        @eps = info[:eps]
      end

      private

      def dtype_name(dtype)
        if dtype.respond_to?(:name)
          dtype.name.to_s
        else
          dtype.to_s
        end
      end
    end

    class Iinfo
      INT_INFO = {
        "bool_" => { min: 0, max: 1 },
        "uint8" => { min: 0, max: 255 },
        "uint16" => { min: 0, max: 65_535 },
        "uint32" => { min: 0, max: 4_294_967_295 },
        "uint64" => { min: 0, max: 18_446_744_073_709_551_615 },
        "int8" => { min: -128, max: 127 },
        "int16" => { min: -32_768, max: 32_767 },
        "int32" => { min: -2_147_483_648, max: 2_147_483_647 },
        "int64" => { min: -9_223_372_036_854_775_808, max: 9_223_372_036_854_775_807 }
      }.freeze

      attr_reader :dtype, :min, :max

      def initialize(dtype)
        @dtype = dtype
        info = INT_INFO[dtype_name(dtype)]
        raise ArgumentError, "unsupported dtype for iinfo: #{dtype_name(dtype)}" if info.nil?

        @min = info[:min]
        @max = info[:max]
      end

      private

      def dtype_name(dtype)
        if dtype.respond_to?(:name)
          dtype.name.to_s
        else
          dtype.to_s
        end
      end
    end

    class ArrayLike
      attr_reader :object

      def initialize(object)
        unless object.respond_to?(:__mlx__array__)
          raise TypeError, "ArrayLike requires an object that responds to __mlx__array__"
        end
        @object = object
      end

      def to_a
        out = @object.__mlx__array__
        raise TypeError, "__mlx__array__ must return MLX::Core::Array" unless out.is_a?(MLX::Core::Array)

        out
      end
    end

    class ArrayIterator
      def initialize(array)
        @array = array
        @index = 0
      end

      def __iter__
        self
      end

      def __next__
        raise StopIteration if @index >= @array.__len__

        out = @array.__getitem__(@index)
        @index += 1
        out
      end

      alias next __next__
    end

    class ArrayAt
      def initialize(array)
        @array = array
        @indices = nil
      end

      def [](indices)
        @indices = indices
        self
      end

      def add(value)
        apply(value) { |lhs, rhs| MLX::Core.add(lhs, rhs) }
      end

      def subtract(value)
        apply(value) { |lhs, rhs| MLX::Core.subtract(lhs, rhs) }
      end

      def multiply(value)
        apply(value) { |lhs, rhs| MLX::Core.multiply(lhs, rhs) }
      end

      def divide(value)
        apply(value) { |lhs, rhs| MLX::Core.divide(lhs, rhs) }
      end

      def maximum(value)
        apply(value) { |lhs, rhs| MLX::Core.maximum(lhs, rhs) }
      end

      def minimum(value)
        apply(value) { |lhs, rhs| MLX::Core.minimum(lhs, rhs) }
      end

      private

      def apply(value)
        raise ArgumentError, "must provide indices to array.at first" if @indices.nil?

        current = @array.__getitem__(@indices)
        rhs = value.is_a?(MLX::Core::Array) ? value : MLX::Core.array(value, current.dtype)
        updated = yield(current, rhs)
        @array.__setitem__(@indices, updated)
      end
    end

    class DLPackCapsule
      attr_reader :array, :dtype, :shape, :device, :stream

      def initialize(array, device:, stream: nil)
        unless array.is_a?(MLX::Core::Array)
          raise TypeError, "DLPackCapsule requires an MLX::Core::Array"
        end

        @array = array
        @dtype = array.dtype
        @shape = array.shape.dup.freeze
        @device = device.dup.freeze
        @stream = stream
      end

      def to_h
        {
          "dtype" => (dtype.respond_to?(:name) ? dtype.name.to_s : dtype.to_s),
          "shape" => shape,
          "device" => device,
          "stream" => stream
        }
      end
    end

    class CustomFunction
      def initialize(fun)
        raise TypeError, "expected callable object" unless fun.respond_to?(:call)

        @fun = fun
        @vjp = nil
        @jvp = nil
        @vmap = nil
      end

      def call(*args, **kwargs, &block)
        @fun.call(*args, **kwargs, &block)
      end

      def vjp(fun = nil, &block)
        @vjp = fun || block
        raise ArgumentError, "expected callable object" unless @vjp.respond_to?(:call)

        @vjp
      end

      def jvp(fun = nil, &block)
        @jvp = fun || block
        raise ArgumentError, "expected callable object" unless @jvp.respond_to?(:call)

        @jvp
      end

      def vmap(fun = nil, &block)
        @vmap = fun || block
        raise ArgumentError, "expected callable object" unless @vmap.respond_to?(:call)

        @vmap
      end

      def custom_vjp?
        !@vjp.nil?
      end

      def custom_jvp?
        !@jvp.nil?
      end

      def custom_vmap?
        !@vmap.nil?
      end

      def call_custom_vjp(primals, cotangents, outputs)
        raise ArgumentError, "custom vjp is not defined" unless custom_vjp?

        @vjp.call(primals, cotangents, outputs)
      end

      def call_custom_jvp(primals, tangents)
        raise ArgumentError, "custom jvp is not defined" unless custom_jvp?

        @jvp.call(primals, tangents)
      end

      def call_custom_vmap(inputs, axes)
        raise ArgumentError, "custom vmap is not defined" unless custom_vmap?

        @vmap.call(inputs, axes)
      end
    end

    class StreamContext
      def initialize(target)
        @target = target
        @previous_device = nil
        @previous_stream = nil
      end

      def enter
        @previous_device = MLX::Core.default_device
        @previous_stream = MLX::Core.default_stream(@previous_device)
        MLX::Core.native_stream(@target)
        self
      end

      def exit(*)
        return self if @previous_device.nil?

        MLX::Core.set_default_device(@previous_device)
        MLX::Core.set_default_stream(@previous_stream)
        @previous_device = nil
        @previous_stream = nil
        self
      end
    end

    PY_EXTRACT_NPZ = <<~PY.freeze
      import os, sys, zipfile
      src = sys.argv[1]
      out_dir = sys.argv[2]
      with zipfile.ZipFile(src, "r") as zf:
          zf.extractall(out_dir)
    PY

    PY_BUILD_NPZ = <<~PY.freeze
      import os, sys, zipfile
      out_path = sys.argv[1]
      in_dir = sys.argv[2]
      compressed = sys.argv[3] == "1"
      mode = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
      with zipfile.ZipFile(out_path, "w", compression=mode, allowZip64=True) as zf:
          for name in sorted(os.listdir(in_dir)):
              zf.write(os.path.join(in_dir, name), arcname=name)
    PY

    module_function

    def ensure_native!
      return if MLX.native_available?

      raise NativeUnavailableError,
            "MLX native extension is unavailable. Build ext/mlx first."
    end

    def available?
      MLX.native_available?
    end

    class << self
      alias_method :native_load, :load if method_defined?(:load)
      alias_method :native_grad, :grad if method_defined?(:grad) && !method_defined?(:native_grad)
      alias_method :native_value_and_grad,
                   :value_and_grad if method_defined?(:value_and_grad) && !method_defined?(:native_value_and_grad)
      alias_method :native_compile, :compile if method_defined?(:compile) && !method_defined?(:native_compile)
      alias_method :native_checkpoint,
                   :checkpoint if method_defined?(:checkpoint) && !method_defined?(:native_checkpoint)
      alias_method :native_stream, :stream if method_defined?(:stream) && !method_defined?(:native_stream)
      alias_method :native_jvp, :jvp if method_defined?(:jvp) && !method_defined?(:native_jvp)
      alias_method :native_vjp, :vjp if method_defined?(:vjp) && !method_defined?(:native_vjp)
      alias_method :native_vmap, :vmap if method_defined?(:vmap) && !method_defined?(:native_vmap)
      alias_method :native_export_to_dot,
                   :export_to_dot if method_defined?(:export_to_dot) && !method_defined?(:native_export_to_dot)

      ARRAY_LEAF = :__mlx_array_leaf__

      def load(file, format = nil, return_metadata = false)
        ensure_native!
        format_name = (format || infer_format(file)).to_s
        if format_name == "npz"
          raise ArgumentError, "metadata not supported for format npz" if return_metadata

          return load_npz(file)
        end

        native_load(file, format, return_metadata)
      end

      def savez(file, *args, **kwargs)
        ensure_native!
        save_npz(file, args, kwargs, false)
      end

      def savez_compressed(file, *args, **kwargs)
        ensure_native!
        save_npz(file, args, kwargs, true)
      end

      def export_to_dot(target, *outputs)
        ensure_native!
        raise ArgumentError, "export_to_dot expects at least one output" if outputs.empty?

        if target.respond_to?(:write)
          Dir.mktmpdir do |dir|
            path = File.join(dir, "graph.dot")
            native_export_to_dot(path, *outputs)
            content = File.binread(path)
            target.write(content)
            target.rewind if target.respond_to?(:rewind)
            content
          end
        else
          native_export_to_dot(target, *outputs)
        end
      end

      def full_like(array, fill_value, dtype = nil)
        ensure_native!
        raise TypeError, "full_like expects an MLX::Core::Array" unless array.is_a?(MLX::Core::Array)

        target_dtype = dtype || array.dtype
        full(array.shape, fill_value, target_dtype)
      end

      def grad(fun, argnums = nil, argnames = nil)
        ensure_native!
        if fun.is_a?(CustomFunction) && fun.custom_vjp?
          return build_custom_vjp_grad_function(fun)
        end

        argnums_v, argnames_v = normalize_diff_targets(argnums, argnames)
        build_grad_like_function(fun, argnums_v, argnames_v, false)
      end

      def value_and_grad(fun, argnums = nil, argnames = nil)
        ensure_native!
        if fun.is_a?(CustomFunction) && fun.custom_vjp?
          return build_custom_vjp_value_and_grad_function(fun)
        end

        argnums_v, argnames_v = normalize_diff_targets(argnums, argnames)
        build_grad_like_function(fun, argnums_v, argnames_v, true)
      end

      def compile(fun, inputs = nil, outputs = nil, shapeless = false)
        ensure_native!
        cache = {}

        lambda do |*args, **kwargs|
          flat_inputs = []
          input_spec = flatten_tree_spec([args, kwargs], flat_inputs, false)
          key = Marshal.dump(input_spec)

          entry = cache[key]
          unless entry
            output_spec = nil
            lifted = lambda do |*flat_vars|
              rebuilt, cursor = inflate_tree_from_arrays(input_spec, flat_vars, 0)
              unless cursor == flat_vars.length
                raise RuntimeError, "internal input reconstruction mismatch"
              end

              call_args = rebuilt[0]
              call_kwargs = rebuilt[1]
              raw_output = fun.call(*call_args, **call_kwargs)

              flat_output = []
              output_spec = flatten_tree_spec(raw_output, flat_output, false)
              flat_output
            end

            compiled = native_compile(lifted, inputs, outputs, shapeless)
            entry = { fn: compiled, output_spec: -> { output_spec } }
            cache[key] = entry
          end

          flat_output = normalize_array_sequence(entry[:fn].call(*flat_inputs), "compiled output")
          spec = entry[:output_spec].call
          raise RuntimeError, "missing output structure from compiled function" if spec.nil?

          rebuilt, cursor = inflate_tree_from_arrays(spec, flat_output, 0)
          unless cursor == flat_output.length
            raise RuntimeError, "internal output reconstruction mismatch"
          end
          rebuilt
        end
      end

      def checkpoint(fun)
        ensure_native!
        cache = {}

        lambda do |*args, **kwargs|
          flat_inputs = []
          input_spec = flatten_tree_spec([args, kwargs], flat_inputs, false)
          key = Marshal.dump(input_spec)

          entry = cache[key]
          unless entry
            output_spec = nil
            lifted = lambda do |*flat_vars|
              rebuilt, cursor = inflate_tree_from_arrays(input_spec, flat_vars, 0)
              unless cursor == flat_vars.length
                raise RuntimeError, "internal input reconstruction mismatch"
              end

              call_args = rebuilt[0]
              call_kwargs = rebuilt[1]
              raw_output = fun.call(*call_args, **call_kwargs)

              flat_output = []
              output_spec = flatten_tree_spec(raw_output, flat_output, false)
              flat_output
            end

            checkpointed = native_checkpoint(lifted)
            entry = { fn: checkpointed, output_spec: -> { output_spec } }
            cache[key] = entry
          end

          flat_output = normalize_array_sequence(entry[:fn].call(*flat_inputs), "checkpoint output")
          spec = entry[:output_spec].call
          raise RuntimeError, "missing output structure from checkpoint function" if spec.nil?

          rebuilt, cursor = inflate_tree_from_arrays(spec, flat_output, 0)
          unless cursor == flat_output.length
            raise RuntimeError, "internal output reconstruction mismatch"
          end
          rebuilt
        end
      end

      def stream(stream_or_device, &block)
        ensure_native!
        if block_given?
          native_stream(stream_or_device, &block)
        else
          StreamContext.new(stream_or_device)
        end
      end

      def jvp(fun, primals, tangents)
        ensure_native!
        if fun.is_a?(CustomFunction) && fun.custom_jvp?
          return custom_jvp(fun, primals, tangents)
        end
        native_jvp(fun, primals, tangents)
      end

      def vjp(fun, primals, cotangents)
        ensure_native!
        if fun.is_a?(CustomFunction) && fun.custom_vjp?
          return custom_vjp(fun, primals, cotangents)
        end
        native_vjp(fun, primals, cotangents)
      end

      def vmap(fun, in_axes = nil, out_axes = nil)
        ensure_native!
        if fun.is_a?(CustomFunction) && fun.custom_vmap?
          return custom_vmap_callable(fun, in_axes, out_axes)
        end
        native_vmap(fun, in_axes, out_axes)
      end

      def custom_function(fun = nil, &block)
        callable = fun || block
        raise ArgumentError, "custom_function requires a callable" if callable.nil?

        CustomFunction.new(callable)
      end

      def finfo(dtype)
        Finfo.new(dtype)
      end

      def iinfo(dtype)
        Iinfo.new(dtype)
      end

      def from_dlpack(dlpack_value)
        case dlpack_value
        when MLX::Core::DLPackCapsule
          source = dlpack_value.array
          array(source.to_a, dlpack_value.dtype)
        when MLX::Core::Array
          array(dlpack_value.to_a, dlpack_value.dtype)
        else
          raise TypeError, "from_dlpack expects MLX::Core::DLPackCapsule or MLX::Core::Array"
        end
      end

      private

      def infer_format(file)
        path = file_path(file)
        ext = File.extname(path).delete_prefix(".")
        raise ArgumentError, "could not infer load format from file extension" if ext.empty?

        ext
      end

      def file_path(file)
        if file.respond_to?(:to_path)
          file.to_path.to_s
        else
          file.to_s
        end
      end

      def python_bin
        ENV.fetch("PYTHON", "python3")
      end

      def run_python!(*argv)
        stdout, stderr, status = Open3.capture3(*argv)
        return if status.success?

        raise RuntimeError, <<~MSG
          python command failed: #{argv.join(" ")}
          stdout:
          #{stdout}
          stderr:
          #{stderr}
        MSG
      end

      def load_npz(file)
        path = file_path(file)
        Dir.mktmpdir("mlx-ruby-npz-load") do |dir|
          run_python!(python_bin, "-c", PY_EXTRACT_NPZ, path, dir)
          out = {}
          Dir.glob(File.join(dir, "**", "*.npy")).sort.each do |npy_path|
            rel = npy_path.delete_prefix(dir + File::SEPARATOR)
            key = rel.end_with?(".npy") ? rel[0...-4] : rel
            out[key] = native_load(npy_path, "npy", false)
          end
          out
        end
      end

      def save_npz(file, args, kwargs, compressed)
        path = file_path(file)
        path = "#{path}.npz" unless path.end_with?(".npz")

        arrays = kwargs.transform_keys(&:to_s)
        args.each_with_index do |value, i|
          key = "arr_#{i}"
          if arrays.key?(key)
            raise ArgumentError, "Cannot use un-named variables and keyword #{key}"
          end
          arrays[key] = value
        end

        Dir.mktmpdir("mlx-ruby-npz-save") do |dir|
          arrays.each do |name, value|
            array_value = value.is_a?(MLX::Core::Array) ? value : MLX::Core.array(value)
            save(File.join(dir, "#{name}.npy"), array_value)
          end
          run_python!(python_bin, "-c", PY_BUILD_NPZ, path, dir, compressed ? "1" : "0")
        end

        nil
      end

      def normalize_diff_targets(argnums, argnames)
        argnames_v = normalize_argnames(argnames)
        argnums_v = normalize_argnums(argnums, argnames_v)
        if argnums_v.empty? && argnames_v.empty?
          raise ArgumentError, "Gradient wrt no argument requested"
        end
        [argnums_v, argnames_v]
      end

      def normalize_argnums(argnums, argnames)
        if argnums.nil?
          return argnames.empty? ? [0] : []
        end
        values = if argnums.is_a?(::Integer)
          [argnums]
        elsif argnums.is_a?(::Array)
          argnums
        else
          raise TypeError, "argnums must be an Integer, an Array of Integer, or nil"
        end
        out = values.map do |value|
          raise TypeError, "argnums entries must be Integer" unless value.is_a?(::Integer)
          raise ArgumentError, "argnums cannot contain negative values" if value.negative?
          value
        end
        raise ArgumentError, "duplicate argnums are not allowed" if out.uniq.length != out.length

        out
      end

      def normalize_argnames(argnames)
        return [] if argnames.nil?
        values = if argnames.is_a?(::String) || argnames.is_a?(::Symbol)
          [argnames]
        elsif argnames.is_a?(::Array)
          argnames
        else
          raise TypeError, "argnames must be a String, Symbol, Array, or nil"
        end
        out = values.map(&:to_s)
        raise ArgumentError, "duplicate argnames are not allowed" if out.uniq.length != out.length

        out
      end

      def build_grad_like_function(fun, argnums, argnames, with_value)
        lambda do |*args, **kwargs|
          selections, flat_inputs = build_target_selections(args, kwargs, argnums, argnames)
          native_argnums = (0...flat_inputs.length).to_a
          lifted = lambda do |*flat_vars|
            call_args, call_kwargs = apply_flat_vars_to_targets(args, kwargs, selections, flat_vars)
            extract_loss(fun.call(*call_args, **call_kwargs))
          end

          if with_value
            native_fn = native_value_and_grad(lifted, native_argnums)
            _loss, raw_grads = native_fn.call(*flat_inputs)
            value = fun.call(*args, **kwargs)
            [value, rebuild_grad_result(raw_grads, selections, argnames)]
          else
            native_fn = native_grad(lifted, native_argnums)
            raw_grads = native_fn.call(*flat_inputs)
            rebuild_grad_result(raw_grads, selections, argnames)
          end
        end
      end

      def build_custom_vjp_grad_function(fun)
        lambda do |*args, **kwargs|
          unless kwargs.empty?
            raise ArgumentError, "custom-function grad currently supports positional arguments only"
          end
          outputs = normalize_array_output(fun.call(*args), "custom_function output")
          cotangents = outputs.map { |out| MLX::Core.ones_like(out) }
          output_arg = outputs.length == 1 ? outputs[0] : outputs
          grads = normalize_array_output(
            fun.call_custom_vjp(args, cotangents, output_arg),
            "custom_function vjp output"
          )
          grads.length == 1 ? grads[0] : grads
        end
      end

      def build_custom_vjp_value_and_grad_function(fun)
        grad_fn = build_custom_vjp_grad_function(fun)
        lambda do |*args, **kwargs|
          value = fun.call(*args, **kwargs)
          [value, grad_fn.call(*args, **kwargs)]
        end
      end

      def custom_jvp(fun, primals, tangents)
        primals_list = normalize_array_output(primals, "primals")
        tangents_list = normalize_array_output(tangents, "tangents")
        outputs = normalize_array_output(fun.call(*primals_list), "custom_function output")
        jvps = normalize_array_output(
          fun.call_custom_jvp(primals_list, tangents_list),
          "custom_function jvp output"
        )
        [outputs, jvps]
      end

      def custom_vjp(fun, primals, cotangents)
        primals_list = normalize_array_output(primals, "primals")
        cotangents_list = normalize_array_output(cotangents, "cotangents")
        outputs = normalize_array_output(fun.call(*primals_list), "custom_function output")
        output_arg = outputs.length == 1 ? outputs[0] : outputs
        vjps = normalize_array_output(
          fun.call_custom_vjp(primals_list, cotangents_list, output_arg),
          "custom_function vjp output"
        )
        [outputs, vjps]
      end

      def custom_vmap_callable(fun, in_axes, _out_axes)
        lambda do |*args|
          input_axes = if in_axes.nil?
            ::Array.new(args.length, 0)
          elsif in_axes.is_a?(::Integer)
            ::Array.new(args.length, in_axes)
          elsif in_axes.is_a?(::Array)
            in_axes
          else
            raise TypeError, "in_axes must be Integer, Array, or nil"
          end
          out = fun.call_custom_vmap(args, input_axes)
          if out.is_a?(::Array) && out.length == 2
            out[0]
          else
            out
          end
        end
      end

      def extract_loss(output)
        return output if output.is_a?(MLX::Core::Array)

        if output.is_a?(::Array) && !output.empty? && output[0].is_a?(MLX::Core::Array)
          return output[0]
        end

        raise ArgumentError,
              "function must return an MLX::Core::Array or an Array whose first element is an MLX::Core::Array"
      end

      def build_target_selections(args, kwargs, argnums, argnames)
        positional = []
        keyword = []
        flat_inputs = []

        argnums.each do |index|
          if index >= args.length
            raise ArgumentError,
                  "Can't compute gradient for positional argument #{index} when #{args.length} positional arguments were provided"
          end
          spec = flatten_tree_spec(args[index], flat_inputs, true)
          positional << { index: index, spec: spec }
        end

        argnames.each do |name|
          key = kwarg_key_for_name(kwargs, name)
          unless key
            raise ArgumentError,
                  "Can't compute gradient for keyword argument '#{name}' because it was not provided"
          end
          spec = flatten_tree_spec(kwargs[key], flat_inputs, true)
          keyword << { key: key, name: name, spec: spec }
        end

        [{ positional: positional, keyword: keyword }, flat_inputs]
      end

      def flatten_tree_spec(value, arrays, strict_arrays)
        if value.is_a?(MLX::Core::Array)
          arrays << value
          return ARRAY_LEAF
        end
        if value.is_a?(::Array)
          return [:array, value.map { |item| flatten_tree_spec(item, arrays, strict_arrays) }]
        end
        if value.is_a?(::Hash)
          return [:hash, value.map { |k, v| [k, flatten_tree_spec(v, arrays, strict_arrays)] }]
        end
        if strict_arrays
          raise TypeError, "[tree_flatten] The argument should contain only arrays"
        end
        if value.nil? || value.is_a?(::Numeric) || value.is_a?(::String) ||
            value.is_a?(::Symbol) || value == true || value == false
          return [:const, value]
        end
        raise TypeError,
              "[compile] Function arguments and outputs must be trees of arrays or constants (Numeric, String, Symbol, true/false, nil)"
      end

      def inflate_tree_from_arrays(spec, arrays, cursor)
        return [arrays.fetch(cursor), cursor + 1] if spec == ARRAY_LEAF

        tag, payload = spec
        case tag
        when :array
          out = []
          payload.each do |child_spec|
            item, cursor = inflate_tree_from_arrays(child_spec, arrays, cursor)
            out << item
          end
          [out, cursor]
        when :hash
          out = {}
          payload.each do |key, child_spec|
            item, cursor = inflate_tree_from_arrays(child_spec, arrays, cursor)
            out[key] = item
          end
          [out, cursor]
        when :const
          [payload, cursor]
        else
          raise ArgumentError, "invalid tree specification"
        end
      end

      def normalize_raw_grads(raw)
        normalize_array_sequence(raw, "gradient")
      end

      def normalize_array_sequence(raw, context)
        return [raw] if raw.is_a?(MLX::Core::Array)

        if raw.is_a?(::Array) && raw.all? { |item| item.is_a?(MLX::Core::Array) }
          return raw
        end
        raise TypeError, "unexpected #{context} return type"
      end

      def normalize_array_output(raw, context)
        if raw.is_a?(MLX::Core::Array)
          [raw]
        elsif raw.is_a?(::Array) && raw.all? { |item| item.is_a?(MLX::Core::Array) }
          raw
        else
          raise TypeError, "unexpected #{context} type"
        end
      end

      def rebuild_grad_result(raw_grads, selections, argnames)
        grad_arrays = normalize_raw_grads(raw_grads)
        cursor = 0

        positional_grads = selections[:positional].map do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], grad_arrays, cursor)
          value
        end
        keyword_grads = {}
        selections[:keyword].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], grad_arrays, cursor)
          keyword_grads[entry[:name]] = value
        end
        unless cursor == grad_arrays.length
          raise RuntimeError, "internal gradient reconstruction mismatch"
        end

        if argnames.empty?
          return positional_grads[0] if positional_grads.length == 1
          return positional_grads
        end

        positional_out = if positional_grads.empty?
          nil
        elsif positional_grads.length == 1
          positional_grads[0]
        else
          positional_grads
        end
        [positional_out, keyword_grads]
      end

      def apply_flat_vars_to_targets(args, kwargs, selections, flat_vars)
        rebuilt_args = args.dup
        rebuilt_kwargs = kwargs.dup
        cursor = 0

        selections[:positional].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], flat_vars, cursor)
          rebuilt_args[entry[:index]] = value
        end

        selections[:keyword].each do |entry|
          value, cursor = inflate_tree_from_arrays(entry[:spec], flat_vars, cursor)
          rebuilt_kwargs[entry[:key]] = value
        end

        unless cursor == flat_vars.length
          raise RuntimeError, "internal target reconstruction mismatch"
        end
        [rebuilt_args, rebuilt_kwargs]
      end

      def kwarg_key_for_name(kwargs, name)
        symbol = name.to_sym
        return symbol if kwargs.key?(symbol)
        return name if kwargs.key?(name)

        nil
      end
    end

    class Device
      alias_method :native_equal, :== if method_defined?(:==) && !method_defined?(:native_equal)

      def ==(other)
        if other.is_a?(::Symbol) || other.is_a?(::String)
          type == other.to_sym
        else
          native_equal(other)
        end
      end

      alias eql? ==
    end

    class Array
      EPSILON_BY_DTYPE = {
        "float16" => 9.765625e-4,
        "bfloat16" => 7.8125e-3,
        "float32" => 1.1920929e-7,
        "float64" => Float::EPSILON,
        "complex64" => 1.1920929e-7
      }.freeze

      def T
        transpose
      end

      def at
        ArrayAt.new(self)
      end

      def real
        MLX::Core.real(self)
      end

      def imag
        MLX::Core.imag(self)
      end

      def itemsize
        dtype.size
      end

      def nbytes
        size * itemsize
      end

      def add(other)
        MLX::Core.add(self, other)
      end

      def subtract(other)
        MLX::Core.subtract(self, other)
      end

      def multiply(other)
        MLX::Core.multiply(self, other)
      end

      def divide(other)
        MLX::Core.divide(self, other)
      end

      def exp
        MLX::Core.exp(self)
      end

      def sin
        MLX::Core.sin(self)
      end

      def cos
        MLX::Core.cos(self)
      end

      def mean(axis = nil)
        MLX::Core.mean(self, axis)
      end

      def sum(axis = nil)
        MLX::Core.sum(self, axis)
      end

      def var(axis = nil, keepdims = nil, ddof = nil)
        MLX::Core.var(self, axis, keepdims, ddof)
      end

      def std(axis = nil, keepdims = nil, ddof = nil)
        MLX::Core.std(self, axis, keepdims, ddof)
      end

      def max(axis = nil, keepdims = nil)
        MLX::Core.max(self, axis, keepdims)
      end

      def min(axis = nil, keepdims = nil)
        MLX::Core.min(self, axis, keepdims)
      end

      def reshape(*shape)
        target = shape.length == 1 ? shape[0] : shape
        MLX::Core.reshape(self, target)
      end

      def transpose(axes = nil)
        MLX::Core.transpose(self, axes)
      end

      def squeeze(axis = nil)
        MLX::Core.squeeze(self, axis)
      end

      def square
        MLX::Core.square(self)
      end

      def sqrt
        MLX::Core.sqrt(self)
      end

      def rsqrt
        MLX::Core.rsqrt(self)
      end

      def reciprocal
        MLX::Core.reciprocal(self)
      end

      def abs
        MLX::Core.abs(self)
      end

      def all(axis = nil, keepdims = nil)
        MLX::Core.all(self, axis, keepdims)
      end

      def any(axis = nil, keepdims = nil)
        MLX::Core.any(self, axis, keepdims)
      end

      def argmax(axis = nil, keepdims = nil)
        MLX::Core.argmax(self, axis, keepdims)
      end

      def argmin(axis = nil, keepdims = nil)
        MLX::Core.argmin(self, axis, keepdims)
      end

      def astype(dtype, stream = nil)
        if stream.nil?
          MLX::Core.astype(self, dtype)
        else
          MLX::Core.astype(self, dtype, stream)
        end
      end

      def conj
        MLX::Core.conj(self)
      end

      def cummax(*args)
        MLX::Core.cummax(self, *args)
      end

      def cummin(*args)
        MLX::Core.cummin(self, *args)
      end

      def cumprod(*args)
        MLX::Core.cumprod(self, *args)
      end

      def cumsum(*args)
        MLX::Core.cumsum(self, *args)
      end

      def diag(*args)
        MLX::Core.diag(self, *args)
      end

      def diagonal(*args)
        MLX::Core.diagonal(self, *args)
      end

      def flatten(start_axis = 0, end_axis = -1)
        MLX::Core.flatten(self, start_axis, end_axis)
      end

      def log
        MLX::Core.log(self)
      end

      def log10
        MLX::Core.log10(self)
      end

      def log1p
        MLX::Core.log1p(self)
      end

      def log2
        MLX::Core.log2(self)
      end

      def logcumsumexp(*args)
        MLX::Core.logcumsumexp(self, *args)
      end

      def logsumexp(*args)
        MLX::Core.logsumexp(self, *args)
      end

      def maximum(other)
        MLX::Core.maximum(self, other)
      end

      def minimum(other)
        MLX::Core.minimum(self, other)
      end

      def moveaxis(source, destination)
        MLX::Core.moveaxis(self, source, destination)
      end

      def prod(axis = nil, keepdims = nil)
        MLX::Core.prod(self, axis, keepdims)
      end

      def round(decimals = 0)
        MLX::Core.round(self, decimals)
      end

      def split(indices_or_sections, axis = 0)
        MLX::Core.split(self, indices_or_sections, axis)
      end

      def swapaxes(axis1, axis2)
        MLX::Core.swapaxes(self, axis1, axis2)
      end

      def view(dtype)
        MLX::Core.view(self, dtype)
      end

      def eps
        dtype_name = if dtype.respond_to?(:name)
          dtype.name.to_s
        else
          dtype.to_s
        end
        EPSILON_BY_DTYPE.fetch(dtype_name, Float::EPSILON)
      end

      def tolist
        to_a
      end

      def __add__(other)
        add(other)
      end

      def __sub__(other)
        subtract(other)
      end

      def __mul__(other)
        multiply(other)
      end

      def __truediv__(other)
        divide(other)
      end

      def __div__(other)
        __truediv__(other)
      end

      def __matmul__(other)
        MLX::Core.matmul(self, other)
      end

      def __imatmul__(other)
        __matmul__(other)
      end

      def __len__
        shape.first || 0
      end

      def __iter__
        ArrayIterator.new(self)
      end

      def __next__
        @__mlx_array_iterator ||= __iter__
        @__mlx_array_iterator.__next__
      end

      def __init__(*_)
        self
      end

      def __repr__
        inspect
      end

      def __bool__
        raise ArgumentError, "The truth value of an array with more than one element is ambiguous" if size != 1

        !!item
      end

      def __int__
        raise ArgumentError, "only size-1 arrays can be converted to Integer" if size != 1

        Integer(item)
      end

      def __float__
        raise ArgumentError, "only size-1 arrays can be converted to Float" if size != 1

        Float(item)
      end

      def __hash__
        object_id.hash
      end

      def __array_namespace__
        MLX::Core
      end

      def __eq__(other)
        MLX::Core.equal(self, other)
      end

      def __ne__(other)
        MLX::Core.not_equal(self, other)
      end

      def __abs__
        MLX::Core.abs(self)
      end

      def __neg__
        MLX::Core.negative(self)
      end

      def __pow__(other)
        MLX::Core.power(self, other)
      end

      def __rpow__(other)
        MLX::Core.power(other, self)
      end

      def __floordiv__(other)
        MLX::Core.floor_divide(self, other)
      end

      def __mod__(other)
        MLX::Core.remainder(self, other)
      end

      def __rmod__(other)
        MLX::Core.remainder(other, self)
      end

      def __radd__(other)
        MLX::Core.add(other, self)
      end

      def __rsub__(other)
        MLX::Core.subtract(other, self)
      end

      def __rmul__(other)
        MLX::Core.multiply(other, self)
      end

      def __rtruediv__(other)
        MLX::Core.divide(other, self)
      end

      def __rdiv__(other)
        __rtruediv__(other)
      end

      def __and__(other)
        MLX::Core.bitwise_and(self, other)
      end

      def __or__(other)
        MLX::Core.bitwise_or(self, other)
      end

      def __xor__(other)
        MLX::Core.bitwise_xor(self, other)
      end

      def __invert__
        MLX::Core.bitwise_invert(self)
      end

      def __lshift__(other)
        MLX::Core.left_shift(self, other)
      end

      def __rshift__(other)
        MLX::Core.right_shift(self, other)
      end

      def __lt__(other)
        MLX::Core.less(self, other)
      end

      def __le__(other)
        MLX::Core.less_equal(self, other)
      end

      def __gt__(other)
        MLX::Core.greater(self, other)
      end

      def __ge__(other)
        MLX::Core.greater_equal(self, other)
      end

      def __iadd__(other)
        __add__(other)
      end

      def __isub__(other)
        __sub__(other)
      end

      def __imul__(other)
        __mul__(other)
      end

      def __itruediv__(other)
        __truediv__(other)
      end

      def __ifloordiv__(other)
        __floordiv__(other)
      end

      def __imod__(other)
        __mod__(other)
      end

      def __ipow__(other)
        __pow__(other)
      end

      def __iand__(other)
        __and__(other)
      end

      def __ior__(other)
        __or__(other)
      end

      def __ixor__(other)
        __xor__(other)
      end

      def __ilshift__(other)
        __lshift__(other)
      end

      def __irshift__(other)
        __rshift__(other)
      end

      def __rfloordiv__(other)
        MLX::Core.floor_divide(other, self)
      end

      def __getitem__(index)
        self[index]
      end

      def __setitem__(index, value)
        copy = __ruby_deep_copy(to_a)
        replacement = value.is_a?(MLX::Core::Array) ? value.to_a : value
        __apply_setitem!(copy, index, replacement)
        MLX::Core.array(copy, dtype)
      end

      def __copy__
        MLX::Core.array(to_a, dtype)
      end

      def __deepcopy__(_memo = nil)
        __copy__
      end

      def __getstate__
        dtype_name = if dtype.respond_to?(:name)
          dtype.name.to_s
        else
          dtype.to_s
        end
        {
          "values" => to_a,
          "dtype" => dtype_name
        }
      end

      def __setstate__(state)
        values = state["values"] || state[:values]
        dtype_name = state["dtype"] || state[:dtype]
        if !dtype_name.nil? && MLX::Core.respond_to?(dtype_name.to_sym)
          MLX::Core.array(values, MLX::Core.public_send(dtype_name.to_sym))
        else
          MLX::Core.array(values)
        end
      end

      def __format__(format_spec = "")
        if size == 1 && !format_spec.to_s.empty?
          kernel = Kernel.format(format_spec, item)
          return kernel
        end
        to_a.to_s
      end

      def __dlpack__(stream = nil)
        unless stream.nil? || stream.is_a?(::Integer)
          raise ArgumentError, "__dlpack__ stream must be nil or Integer"
        end

        MLX::Core::DLPackCapsule.new(self, device: __dlpack_device, stream: stream)
      end

      def __dlpack_device
        device = MLX::Core.default_device
        type_id = case device.type
        when :cpu
          1
        when :gpu
          MLX::Core.metal_is_available ? 8 : 13
        else
          device.type
        end
        [type_id, device.index]
      end

      alias __dlpack_device__ __dlpack_device

      private

      def __apply_setitem!(data, index, replacement)
        if index.is_a?(::Integer)
          data[index] = replacement
          return
        end

        normalized = if index.is_a?(MLX::Core::Array)
          index.to_a
        elsif index.is_a?(::Array)
          index
        else
          raise ArgumentError, "__setitem__ supports Integer, Integer list, or boolean mask indices"
        end

        unless data.is_a?(::Array)
          raise ArgumentError, "__setitem__ list/mask indices require array values"
        end

        if normalized.all? { |v| v == true || v == false }
          __apply_boolean_mask_setitem!(data, normalized, replacement)
          return
        end

        unless normalized.all? { |v| v.is_a?(::Integer) }
          raise ArgumentError, "__setitem__ list indices must be all Integers or all booleans"
        end

        __apply_integer_list_setitem!(data, normalized, replacement)
      end

      def __apply_boolean_mask_setitem!(data, mask, replacement)
        if mask.length != data.length
          raise ArgumentError, "__setitem__ boolean mask must match array length"
        end

        replacement_values = replacement.is_a?(::Array) ? replacement.flatten : nil
        replacement_index = 0

        mask.each_with_index do |flag, i|
          next unless flag

          if replacement_values
            if replacement_index >= replacement_values.length
              raise ArgumentError, "__setitem__ replacement values shorter than mask true count"
            end
            data[i] = replacement_values[replacement_index]
            replacement_index += 1
          else
            data[i] = replacement
          end
        end
      end

      def __apply_integer_list_setitem!(data, indices, replacement)
        if replacement.is_a?(::Array)
          values = replacement.flatten
          if values.length == 1
            indices.each { |i| data[i] = values[0] }
            return
          end
          if values.length != indices.length
            raise ArgumentError, "__setitem__ replacement values must match index list length"
          end

          indices.each_with_index { |i, offset| data[i] = values[offset] }
        else
          indices.each { |i| data[i] = replacement }
        end
      end

      def __ruby_deep_copy(value)
        return value.map { |item| __ruby_deep_copy(item) } if value.is_a?(::Array)

        value
      end
    end
  end
end
