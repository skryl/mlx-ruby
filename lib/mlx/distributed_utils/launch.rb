# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "pathname"
require "rbconfig"
require "shellwords"
require "tempfile"

module MLX
  module DistributedUtils
    class CommandProcess
      def process
        raise NotImplementedError
      end

      def exit_status
        raise NotImplementedError
      end

      def preprocess_output(_data, is_stdout: false)
        _ = is_stdout
        raise NotImplementedError
      end

      def terminate
        raise NotImplementedError
      end
    end

    class RemoteProcess < CommandProcess
      attr_reader :process

      class Handle
        attr_reader :stdin, :stdout, :stderr

        def initialize(command)
          @stdin, @stdout, @stderr, @wait_thr = Open3.popen3("/bin/bash", "-lc", command)
        end

        def poll
          return nil if @wait_thr.alive?

          normalize_status(@wait_thr.value)
        end

        def wait
          normalize_status(@wait_thr.value)
        end

        def terminate
          Process.kill("TERM", @wait_thr.pid)
        rescue Errno::ESRCH
          nil
        end

        private

        def normalize_status(process_status)
          return process_status.exitstatus unless process_status.exitstatus.nil?

          termsig = process_status.termsig
          termsig.nil? ? nil : -termsig
        end
      end

      def initialize(rank, host, python, cwd, files, env, command)
        _ = python
        @host = host
        @is_local = host == "127.0.0.1"
        @pidfile = nil
        @killed = false
        @pid_buffer = +""
        @launch_script = self.class.make_launch_script(rank, cwd, files, env, command, @is_local)
        wrapped = @is_local ? @launch_script : "ssh -tt -o LogLevel=QUIET #{@host} #{Shellwords.escape(@launch_script)}"
        @process = Handle.new(wrapped)
      end

      def exit_status
        return [nil, nil] if @process.nil?

        [@process.poll, @killed]
      end

      def preprocess_output(data, is_stdout: false)
        _ = is_stdout
        if @pidfile.nil?
          @pid_buffer << data
          return "" unless @pid_buffer.include?("\n")

          pidfile, rest = @pid_buffer.split("\n", 2)
          @pidfile = pidfile.to_s.strip
          @pid_buffer = +""
          rest.to_s
        else
          data
        end
      end

      def terminate
        return if @killed

        @process.terminate unless @process.nil?
        @process.wait unless @process.nil?
        if @pidfile.nil? || @pidfile.empty?
          @killed = true
          return
        end

        cmd = self.class.make_kill_script(@pidfile)
        cmd = "ssh #{@host} #{Shellwords.escape(cmd)}" unless @is_local
        stdout, = Open3.capture3("/bin/bash", "-lc", cmd)
        @killed = stdout.strip == "1"
      rescue StandardError
        @killed = true
      end

      def self.make_launch_script(rank, cwd, files, env, command, is_local)
        script = +""
        script << "stty -echo; " unless is_local
        script << "pidfile=$(mktemp); "
        script << "echo $$ > $pidfile; "
        script << "printf \"%s\\n\" $pidfile; "

        d = cwd || Dir.pwd
        script << "if [[ -d #{d.inspect} ]]; then cd #{d.inspect}; else echo 'Failed to change directory to' #{d.inspect} >2; fi; "

        env.each do |entry|
          key, value = entry.split("=", 2)
          next unless key.match?(/\A[a-zA-Z_][a-zA-Z0-9_]*\z/)

          escaped = value.nil? ? "" : Shellwords.escape(value)
          script << "export #{key}=#{escaped}; "
        end

        files.each do |env_name, content|
          script << "fname=$(mktemp); "
          script << "echo #{Shellwords.escape(content.to_s)} >$fname; "
          script << "export #{env_name}=$fname; "
        end

        script << "export MLX_RANK=#{rank}; "
        command_str = command.map { |arg| Shellwords.escape(arg.to_s) }.join(" ")
        script << "cmd=(#{command_str}); "
        script << "exec \"${cmd[@]}\""
        script
      end

      def self.make_kill_script(pidfile)
        script = +""
        script << "pid=$(cat #{pidfile}); "
        script << "if ps -p $pid >/dev/null; then "
        script << "    kill $pid; "
        script << "    echo 1; "
        script << "else "
        script << "    echo 0; "
        script << "fi; "
        script << "rm #{pidfile}"
        script
      end
    end

    class << self
      def launch_ring(parser, hosts, args, command)
        if hosts.any? { |h| h.ips.nil? || h.ips.empty? }
          parser_error(parser, "The ring backend requires IPs to be provided instead of hostnames")
        end

        port = launch_arg(args, :starting_port, 32_323).to_i
        connections_per_ip = launch_arg(args, :connections_per_ip, 1).to_i
        ring_hosts = []
        hosts.each do |h|
          node = []
          h.ips.each do |ip|
            connections_per_ip.times do
              node << "#{ip}:#{port}"
              port += 1
            end
          end
          ring_hosts << node
        end

        files = { "MLX_HOSTFILE" => (ring_hosts.length > 1 ? JSON.dump(ring_hosts) : "") }
        env = Array(launch_arg(args, :env, [])).dup
        env << "MLX_RING_VERBOSE=1" if launch_arg(args, :verbose, false)
        cwd = launch_arg(args, :cwd, nil)
        python = launch_arg(args, :python, RbConfig.ruby)

        log(launch_arg(args, :verbose, false), "Running #{command.map(&:to_s).join(' ')}")
        launch_with_io(
          RemoteProcess,
          hosts.each_with_index.map do |h, rank|
            [[rank, h.ssh_hostname, python, cwd, files, env, command], {}]
          end,
          launch_arg(args, :verbose, false)
        )
      end

      def launch_nccl(_parser, hosts, args, command)
        if hosts.empty? || hosts.first.ips.nil? || hosts.first.ips.empty?
          raise ArgumentError, "Rank 0 should have an IP reachable from all other ranks"
        end

        master_host = hosts.first.ips.first
        master_port = launch_arg(args, :nccl_port, 12_345).to_i
        world_size = hosts.length
        env = Array(launch_arg(args, :env, [])).dup
        env << "NCCL_DEBUG=INFO" if launch_arg(args, :verbose, false)
        env << "NCCL_HOST_IP=#{master_host}"
        env << "NCCL_PORT=#{master_port}"
        env << "MLX_WORLD_SIZE=#{world_size}"

        repeat_hosts = launch_arg(args, :repeat_hosts, 1).to_i
        cwd = launch_arg(args, :cwd, nil)
        python = launch_arg(args, :python, RbConfig.ruby)

        log(launch_arg(args, :verbose, false), "Running #{command.map(&:to_s).join(' ')}")
        launch_with_io(
          RemoteProcess,
          hosts.each_with_index.map do |h, rank|
            [
              [
                rank,
                h.ssh_hostname,
                python,
                cwd,
                {},
                env + ["CUDA_VISIBLE_DEVICES=#{rank % repeat_hosts}"],
                command
              ],
              {}
            ]
          end,
          launch_arg(args, :verbose, false)
        )
      end

      def launch_jaccl(_parser, hosts, args, command)
        if hosts.empty? || hosts.first.ips.nil? || hosts.first.ips.empty?
          raise ArgumentError, "Rank 0 should have an IP reachable from all other ranks"
        end

        have_rdmas = hosts.all? { |h| h.rdma.is_a?(Array) && h.rdma.length == hosts.length }
        have_nulls = hosts.each_with_index.all? { |h, i| h.rdma[i].nil? }
        raise ArgumentError, "Malformed hostfile for jaccl backend" unless have_rdmas && have_nulls

        env = Array(launch_arg(args, :env, [])).dup
        env << "MLX_JACCL_COORDINATOR=#{hosts.first.ips.first}:#{launch_arg(args, :starting_port, 32_323).to_i}"
        env << "MLX_JACCL_RING=1" if launch_arg(args, :backend, "").to_s == "jaccl-ring"
        files = { "MLX_IBV_DEVICES" => JSON.dump(hosts.map(&:rdma)) }
        cwd = launch_arg(args, :cwd, nil)
        python = launch_arg(args, :python, RbConfig.ruby)

        log(launch_arg(args, :verbose, false), "Running #{command.map(&:to_s).join(' ')}")
        launch_with_io(
          RemoteProcess,
          hosts.each_with_index.map do |h, rank|
            [[rank, h.ssh_hostname, python, cwd, files, env, command], {}]
          end,
          launch_arg(args, :verbose, false)
        )
      end

      def get_mpi_libname
        ompi_info = find_executable("ompi_info")
        return nil if ompi_info.nil?

        checker = if RbConfig::CONFIG["host_os"].downcase.include?("darwin")
          ["otool", "-L", ompi_info]
        else
          ["ldd", ompi_info]
        end
        stdout, _stderr, status = Open3.capture3(*checker)
        return nil unless status.success?

        line = stdout.each_line.find { |entry| entry.include?("libmpi") }
        return nil if line.nil?

        line.strip.split.first.to_s.sub(/\A@rpath\//, "")
      rescue StandardError
        nil
      end

      def launch_mpi(_parser, hosts, args, command)
        mpirun = find_executable("mpirun")
        raise ArgumentError, "mpirun is not available in PATH" if mpirun.nil?

        env = Array(launch_arg(args, :env, [])).dup
        mpi_libname = get_mpi_libname
        if !mpi_libname.nil? && !mpi_libname.empty?
          dyld = Pathname.new(mpirun).dirname.dirname.join("lib")
          env = ["DYLD_LIBRARY_PATH=#{dyld}", "MLX_MPI_LIBNAME=#{mpi_libname}"] + env
        end

        host_counts = Hash.new(0)
        hosts.each { |h| host_counts[h.ssh_hostname] += 1 }

        Tempfile.create(["mlx-hosts", ".txt"]) do |file|
          host_counts.each { |h, n| file.puts("#{h} slots=#{n}") }
          file.flush

          cmd = [mpirun, "--output", ":raw", "--hostfile", file.path]
          cwd = launch_arg(args, :cwd, nil)
          cmd += ["-cwd", cwd] unless cwd.nil? || cwd.empty?
          env.each { |entry| cmd += ["-x", entry] }
          Array(launch_arg(args, :mpi_arg, [])).each { |entry| cmd += Shellwords.split(entry.to_s) }
          cmd += ["--"] + command

          log(launch_arg(args, :verbose, false), "Running #{cmd.join(' ')}")
          _out, _err, status = Open3.capture3(*cmd)
          status.exitstatus
        end
      end

      def main(argv = ARGV)
        opts = {
          print_python: false,
          verbose: false,
          hosts: "127.0.0.1",
          repeat_hosts: 1,
          hostfile: nil,
          backend: nil,
          env: [],
          mpi_arg: [],
          connections_per_ip: 1,
          starting_port: 32_323,
          cwd: nil,
          nccl_port: 12_345,
          verify_script: true,
          python: RbConfig.ruby
        }

        parser = OptionParser.new do |o|
          o.on("--print-python") { opts[:print_python] = true }
          o.on("--verbose") { opts[:verbose] = true }
          o.on("--hosts HOSTS") { |v| opts[:hosts] = v }
          o.on("--repeat-hosts N", Integer) { |v| opts[:repeat_hosts] = v }
          o.on("-n N", Integer) { |v| opts[:repeat_hosts] = v }
          o.on("--hostfile FILE") { |v| opts[:hostfile] = v }
          o.on("--backend BACKEND") { |v| opts[:backend] = v }
          o.on("--env ENV") { |v| opts[:env] << v }
          o.on("--mpi-arg ARG") { |v| opts[:mpi_arg] << v }
          o.on("--connections-per-ip N", Integer) { |v| opts[:connections_per_ip] = v }
          o.on("--starting-port PORT", Integer) { |v| opts[:starting_port] = v }
          o.on("-p PORT", Integer) { |v| opts[:starting_port] = v }
          o.on("--cwd DIR") { |v| opts[:cwd] = v }
          o.on("--nccl-port PORT", Integer) { |v| opts[:nccl_port] = v }
          o.on("--no-verify-script") { opts[:verify_script] = false }
          o.on("--python PATH") { |v| opts[:python] = v }
        end
        rest_argv = argv.dup
        parser.order!(rest_argv)
        rest = rest_argv

        if opts[:print_python]
          puts opts[:python]
          return 0
        end
        parser_error(parser, "No script is provided") if rest.empty?
        rest.shift if rest.first == "--"

        hostfile = if opts[:hostfile]
          Hostfile.from_file(opts[:hostfile])
        else
          Hostfile.from_list(opts[:hosts], opts[:repeat_hosts])
        end

        opts[:backend] = hostfile.backend if opts[:backend].nil? && !hostfile.backend.to_s.empty?
        if opts[:backend].nil?
          opts[:backend] = MLX::Core.distributed_is_available("nccl") ? "nccl" : "ring"
        end
        opts[:env] = hostfile.envs + opts[:env]

        command = rest.dup
        script = Pathname.new(command.first)
        if script.file?
          command[0] = opts[:python]
          command.insert(1, script.realpath.to_s)
        elsif (resolved = find_executable(command.first))
          command[0] = resolved
        elsif opts[:verify_script]
          raise ArgumentError, "Invalid script or command #{command.first}"
        end

        args = opts
        case opts[:backend]
        when "ring"
          launch_ring(parser, hostfile.hosts, args, command)
        when "mpi"
          launch_mpi(parser, hostfile.hosts, args, command)
        when "nccl"
          launch_nccl(parser, hostfile.hosts, args, command)
        when "jaccl", "jaccl-ring"
          launch_jaccl(parser, hostfile.hosts, args, command)
        else
          parser_error(parser, "The backend should be one of {'ring', 'mpi', 'nccl', 'jaccl', 'jaccl-ring'}")
        end
        0
      end

      private

      def launch_with_io(command_class, arguments, verbose)
        stop = false
        lock = Mutex.new
        exit_codes = Array.new(arguments.length) { [nil, nil] }

        workers = arguments.each_with_index.map do |(args, kwargs), rank|
          Thread.new do
            command = command_class.new(*args, **kwargs)
            proc = command.process

            stdout_thread = Thread.new do
              stream_output(proc.stdout, $stdout, command, true)
            end
            stderr_thread = Thread.new do
              stream_output(proc.stderr, $stderr, command, false)
            end

            loop do
              break unless proc.poll.nil?
              if lock.synchronize { stop }
                command.terminate
                break
              end
              sleep 0.05
            end

            stdout_thread.join
            stderr_thread.join
            exit_codes[rank] = command.exit_status

            status, killed = exit_codes[rank]
            if killed
              log_warning("Node with rank #{rank} was killed")
            elsif !status.nil? && status != 0
              log_warning("Node with rank #{rank} exited with code #{status}")
              lock.synchronize { stop = true }
            else
              log(verbose, "Node with rank #{rank} completed")
            end
          end
        end

        workers.each(&:join)
        exit_codes
      end

      def stream_output(io, output, command, is_stdout)
        loop do
          chunk = io.readpartial(8192)
          text = command.preprocess_output(chunk, is_stdout: is_stdout)
          next if text.nil? || text.empty?

          output.write(text)
          output.flush
        end
      rescue EOFError, IOError
        nil
      end

      def parser_error(parser, message)
        if parser.respond_to?(:error)
          parser.error(message)
        else
          raise ArgumentError, message
        end
      end

      def launch_arg(args, key, default = nil)
        if args.respond_to?(key)
          args.public_send(key)
        elsif args.respond_to?(:[])
          args[key]
        else
          default
        end || default
      end

      def find_executable(name)
        ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).each do |entry|
          path = File.join(entry, name.to_s)
          return path if File.file?(path) && File.executable?(path)
        end
        nil
      end
    end
  end
end
