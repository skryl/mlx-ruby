# frozen_string_literal: true

require "json"
require "open3"
require "optparse"
require "set"
require "shellwords"

module MLX
  module DistributedUtils
    SSHInfo = Struct.new(:can_ssh, :has_sudo, keyword_init: true) do
      def to_bool
        can_ssh
      end
    end
    ThunderboltPort = Struct.new(:iface, :uuid, :connected_to, keyword_init: true)
    ThunderboltHost = Struct.new(:name, :ports, keyword_init: true)

    class IPConfigurator
      attr_reader :ips, :hosts, :tb_hosts

      def initialize(hosts = [], tb_hosts = [], uuid_reverse_index = {})
        assigned = Set.new
        ip_map = Hash.new { |h, key| h[key] = [] }
        ip0 = 0
        ip1 = 0

        tb_hosts.each_with_index do |host, src_node|
          host.ports.each_with_index do |port, src_port|
            next if port.connected_to.nil?
            next unless uuid_reverse_index.key?(port.connected_to)
            next if assigned.include?([src_node, src_port])

            dst_node, dst_port = uuid_reverse_index[port.connected_to]
            ip_src = "192.168.#{ip0}.#{ip1 + 1}"
            ip_dst = "192.168.#{ip0}.#{ip1 + 2}"
            iface_src = port.iface
            iface_dst = tb_hosts[dst_node].ports[dst_port].iface

            ip_map[[src_node, dst_node]] << [iface_src, ip_src]
            ip_map[[dst_node, src_node]] << [iface_dst, ip_dst]

            assigned.add([src_node, src_port])
            assigned.add([dst_node, dst_port])

            ip1 += 4
            if ip1 > 255
              ip0 += 1
              ip1 = 0
            end
            raise ArgumentError, "Ran out of available local IPs" if ip0 > 255
          end
        end

        @ips = ip_map
        @hosts = hosts
        @tb_hosts = tb_hosts
      end

      def setup(verbose: false, auto_setup: false, runner: nil, input: $stdin, output: $stdout)
        netmask = "255.255.255.252"
        @hosts.each_with_index do |host, i|
          command = +""
          command << "sudo ifconfig bridge0 down\n"
          @hosts.length.times do |j|
            next if i == j
            next unless @ips.key?([i, j])

            @ips[[i, j]].zip(@ips[[j, i]]).each do |(iface, ip), (_peer_iface, peer_ip)|
              command << "sudo ifconfig #{iface} inet #{ip} netmask #{netmask}\n"
              command << "sudo route change #{peer_ip} -interface #{iface}\n"
            end
          end

          if auto_setup
            output.puts("Running auto setup for #{host.ssh_hostname}")
            ssh_cmd = ["ssh", host.ssh_hostname, command.strip.gsub("\n", " ; ")]
            MLX::DistributedUtils.log(verbose, Shellwords.join(ssh_cmd))
            MLX::DistributedUtils.__send__(:run_command, ssh_cmd, runner: runner)
          else
            title = "Setup for #{host.ssh_hostname}"
            output.puts(title)
            output.puts("=" * title.length)
            output.puts(command)
            output.print("Enter to continue")
            input.gets
          end
          output.puts
        end
      end
    end

    class << self
      def add_ips(hosts, verbose: false, runner: nil)
        hosts.each do |host|
          log(verbose, "Getting the ip from", host.ssh_hostname)
          ip = read_ip(host.ssh_hostname, "en0", runner: runner)
          ip = read_ip(host.ssh_hostname, "en1", runner: runner) if ip.empty?

          if ip.empty?
            log_warning("Could not extract ip for", host.ssh_hostname)
          else
            host.ips << ip
          end
        end
      end

      def save_hostfile(args, hostfile)
        if cfg_arg(args, :output_hostfile, nil)
          File.write(cfg_arg(args, :output_hostfile, nil), JSON.pretty_generate(hostfile.to_json_obj))
        else
          puts "Hostfile"
          puts "========"
          puts JSON.pretty_generate(hostfile.to_json_obj)
        end
      end

      def check_rdma(hosts, verbose: false, strict: true, runner: nil)
        failed = false
        hosts.each do |host|
          log(verbose, "Checking that", host.ssh_hostname, "supports RDMA")
          result = run_command(["ssh", host.ssh_hostname, "ibv_devices"], runner: runner)
          devices = stdout_for(result).split.map { |entry| entry.strip }
          rdma = devices.select { |entry| entry.start_with?("rdma_") }
          next unless rdma.empty?

          (strict ? method(:log_error) : method(:log_warning)).call(
            host.ssh_hostname,
            "does not seem to have RDMA enabled"
          )
          failed = true
        end

        if failed
          logger = strict ? method(:log_error) : method(:log_warning)
          logger.call
          logger.call("Some of the hosts don't have RDMA enabled or they don't support RDMA.")
          logger.call
          logger.call("See https://ml-explore.github.io/mlx/build/html/usage/distributed.html")
          logger.call("for instructions on how to enable RDMA.")
        end

        raise ArgumentError, "RDMA validation failed for one or more hosts" if failed && strict

        !failed
      end

      def can_auto_setup(hosts, sshinfo, auto_setup: false)
        has_sudo = sshinfo.all?(&:has_sudo)
        if !has_sudo && auto_setup
          log_warning("Automatic setup requested but some hosts do not have passwordless sudo")
          hosts.zip(sshinfo).each do |host, info|
            log_warning(" - #{host.ssh_hostname}") unless info.has_sudo
          end
        end
        has_sudo
      end

      def parse_hardware_ports(ports_string)
        ports = {}
        port_name = nil
        ports_string.to_s.split("\n").each do |line|
          if line.start_with?("Hardware Port:")
            port_name = line.strip[15..]
          elsif line.start_with?("Device:")
            ports[port_name] = line.strip[8..]
            port_name = nil
          end
        end
        ports
      end

      def extract_connectivity(hosts, verbose, runner: nil)
        thunderbolt_connections = []
        hosts.each do |host|
          log(verbose, "Getting connectivity from", host.ssh_hostname)
          result = run_command(
            ["ssh", host.ssh_hostname, "system_profiler", "SPThunderboltDataType", "-json"],
            runner: runner
          )
          thunderbolt_connections << JSON.parse(stdout_for(result))
        end

        interface_maps = []
        hosts.each do |host|
          log(verbose, "Getting interface names from", host.ssh_hostname)
          result = run_command(
            ["ssh", host.ssh_hostname, "networksetup", "-listallhardwareports"],
            runner: runner
          )
          interface_maps << parse_hardware_ports(stdout_for(result))
        end

        tb_hosts = []
        thunderbolt_connections.zip(interface_maps).each do |connection, iface_map|
          name = ""
          ports = []
          Array(connection["SPThunderboltDataType"]).each do |entry|
            uuid = entry["domain_uuid_key"]
            next if uuid.nil?

            name = entry["device_name_key"].to_s
            tag = entry.dig("receptacle_1_tag", "receptacle_id_key")
            items = Array(entry["_items"])
            connected = items.find { |item| item.key?("domain_uuid_key") }
            connected_to = connected.nil? ? nil : connected["domain_uuid_key"]
            iface = iface_map["Thunderbolt #{tag}"]
            ports << ThunderboltPort.new(iface: iface, uuid: uuid, connected_to: connected_to)
          end
          tb_hosts << ThunderboltHost.new(name: name, ports: ports.sort_by { |port| port.iface.to_s })
        end

        uuid_reverse_index = {}
        tb_hosts.each_with_index do |host, i|
          host.ports.each_with_index do |port, j|
            uuid_reverse_index[port.uuid] = [i, j]
          end
        end

        [tb_hosts, uuid_reverse_index]
      end

      def make_connectivity_matrix(tb_hosts, uuid_reverse_index)
        connectivity = []
        tb_hosts.each_with_index do |host, i|
          row = Array.new(tb_hosts.length, 0)
          host.ports.each do |port|
            next unless uuid_reverse_index.key?(port.connected_to)

            j, = uuid_reverse_index[port.connected_to]
            row[j] += 1
          end
          connectivity[i] = row
        end
        connectivity
      end

      def tb_connectivity_to_dot(hosts, tb_hosts, uuid_reverse_index, io: $stdout)
        names = []
        tb_hosts.length.times do |i|
          value = +""
          j = i
          loop do
            value << (97 + (j % 26)).chr
            j /= 26
            break if j.zero?
          end
          names << value
        end

        io.puts("graph G {")
        io.puts("  node [shape=rectangle];")
        hosts.each_with_index do |host, i|
          io.puts("  #{names[i]} [label=\"#{host.ssh_hostname}\"];")
        end
        tb_hosts.each_with_index do |host, i|
          host.ports.each do |port|
            next if port.connected_to.nil?

            dst = uuid_reverse_index[port.connected_to]
            next if dst.nil? || dst[0] < i

            io.puts("  #{names[i]} -- #{names[dst[0]]} [label=\"#{port.iface}/#{tb_hosts[dst[0]].ports[dst[1]].iface}\"]")
          end
        end
        io.puts("}")
      end

      def extract_rings(connectivity)
        rings = []
        existing = Set.new
        num_nodes = connectivity.length

        dfs = lambda do |start_node, node, path, visited, &blk|
          path << node
          visited.add(node)
          num_nodes.times do |j|
            next unless connectivity[node][j] > 0

            if j == start_node
              blk.call(path.dup)
            elsif !visited.include?(j)
              dfs.call(start_node, j, path, visited, &blk)
            end
          end
          path.pop
          visited.delete(node)
        end

        num_nodes.times do |start|
          dfs.call(start, start, [], Set.new) do |ring|
            count = ring.length.times.map do |i|
              connectivity[ring[i]][ring[(i + 1) % ring.length]]
            end.min
            key = ring.sort
            next if existing.include?(key)

            rings << [ring, count]
            existing.add(key)
          end
        end

        rings.sort_by { |ring, _count| -ring.length }
      end

      def check_valid_mesh(hosts, connectivity, strict: true)
        num_nodes = connectivity.length
        num_nodes.times do |i|
          num_nodes.times do |j|
            next if i == j
            next if connectivity[i][j] > 0

            if strict
              raise ArgumentError,
                    "Incomplete mesh, #{hosts[i].ssh_hostname} is not connected to #{hosts[j].ssh_hostname}"
            else
              return false
            end
          end
        end
        true
      end

      def check_valid_ring(hosts, rings, strict: true)
        has_ring = !rings.empty? && rings[0][0].length == hosts.length
        raise ArgumentError, "Could not find a full ring." if strict && !has_ring

        has_ring
      end

      def check_ssh_connections(hosts, runner: nil)
        infos = hosts.map do |host|
          ssh_result = run_command(
            [
              "ssh",
              "-o",
              "BatchMode=yes",
              "-o",
              "ConnectTimeout=5",
              host.ssh_hostname,
              "echo",
              "success"
            ],
            runner: runner
          )
          can_ssh = success_for(ssh_result)
          has_sudo = false

          if can_ssh
            sudo_result = run_command(
              [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                host.ssh_hostname,
                "sudo",
                "ls"
              ],
              runner: runner
            )
            has_sudo = success_for(sudo_result)
          end

          SSHInfo.new(can_ssh: can_ssh, has_sudo: has_sudo)
        end

        unless infos.all?(&:can_ssh)
          bad_hosts = hosts.zip(infos).filter_map { |host, info| host.ssh_hostname unless info.can_ssh }
          raise ArgumentError, "Could not ssh to the following hosts: #{bad_hosts.join(', ')}"
        end

        infos
      end

      def prepare_ethernet_hostfile(args, hosts, runner: nil)
        log(cfg_arg(args, :verbose, false), "Preparing an ethernet hostfile")
        add_ips(hosts, verbose: cfg_arg(args, :verbose, false), runner: runner)
        hostfile_hosts = hosts.each_with_index.map do |host, i|
          Host.new(rank: i, ssh_hostname: host.ssh_hostname, ips: host.ips, rdma: [])
        end
        hostfile = Hostfile.new(hostfile_hosts, "", cfg_arg(args, :env, []))
        save_hostfile(args, hostfile)
      end

      def configure_ring(args, hosts, ips, ring, sshinfo, runner: nil)
        log(cfg_arg(args, :verbose, false), "Prepare a ring hostfile")
        ring_nodes, count = ring
        ring_hosts = []
        ring_nodes.each_with_index do |node, i|
          host = hosts[node]
          peer = ring_nodes[i - 1]
          peer_ips = count.times.map { |c| ips.ips[[node, peer]][c][1] }
          ring_hosts << Host.new(rank: i, ssh_hostname: host.ssh_hostname, ips: peer_ips, rdma: [])
        end
        hostfile = Hostfile.new(ring_hosts, "ring", cfg_arg(args, :env, []))

        has_sudo = can_auto_setup(hosts, sshinfo, auto_setup: cfg_arg(args, :auto_setup, false))
        ips.setup(
          verbose: cfg_arg(args, :verbose, false),
          auto_setup: cfg_arg(args, :auto_setup, false) && has_sudo,
          runner: runner
        )
        save_hostfile(args, hostfile)
      end

      def configure_jaccl(args, hosts, ips, sshinfo, runner: nil)
        log(cfg_arg(args, :verbose, false), "Prepare a jaccl hostfile")
        add_ips(hosts, verbose: cfg_arg(args, :verbose, false), runner: runner)

        jaccl_hosts = hosts.each_with_index.map do |host, i|
          rdma = hosts.each_index.map do |j|
            if i == j
              nil
            else
              "rdma_#{ips.ips[[i, j]][0][0]}"
            end
          end
          Host.new(rank: i, ssh_hostname: host.ssh_hostname, ips: host.ips, rdma: rdma)
        end
        hostfile = Hostfile.new(jaccl_hosts, "jaccl", cfg_arg(args, :env, []))

        has_sudo = can_auto_setup(hosts, sshinfo, auto_setup: cfg_arg(args, :auto_setup, false))
        ips.setup(
          verbose: cfg_arg(args, :verbose, false),
          auto_setup: cfg_arg(args, :auto_setup, false) && has_sudo,
          runner: runner
        )
        save_hostfile(args, hostfile)
      end

      def configure_jaccl_ring(args, hosts, ips, ring, sshinfo, runner: nil)
        log(cfg_arg(args, :verbose, false), "Prepare a jaccl-ring hostfile")
        add_ips(hosts, verbose: cfg_arg(args, :verbose, false), runner: runner)

        ring_nodes, count = ring
        num_nodes = hosts.length
        jaccl_hosts = []
        ring_nodes.each_with_index do |node, i|
          host = hosts[node]
          peer_left = ring_nodes[i - 1]
          peer_right = ring_nodes[(i + 1) % num_nodes]
          rdmas = hosts.each_index.map do |j|
            if j != peer_left && j != peer_right
              nil
            else
              rdma_values = count.times.map { |c| "rdma_#{ips.ips[[i, j]][c][0]}" }
              count == 1 ? rdma_values[0] : rdma_values
            end
          end
          jaccl_hosts << Host.new(rank: i, ssh_hostname: host.ssh_hostname, ips: host.ips, rdma: rdmas)
        end
        hostfile = Hostfile.new(jaccl_hosts, "jaccl-ring", cfg_arg(args, :env, []))

        has_sudo = can_auto_setup(hosts, sshinfo, auto_setup: cfg_arg(args, :auto_setup, false))
        ips.setup(
          verbose: cfg_arg(args, :verbose, false),
          auto_setup: cfg_arg(args, :auto_setup, false) && has_sudo,
          runner: runner
        )
        save_hostfile(args, hostfile)
      end

      def prepare_tb_hostfile(args, hosts, sshinfo, runner: nil)
        log(cfg_arg(args, :verbose, false), "Preparing for communication over thunderbolt")
        tb_hosts, uuid_reverse_index = extract_connectivity(hosts, cfg_arg(args, :verbose, false), runner: runner)

        if cfg_arg(args, :dot, false)
          tb_connectivity_to_dot(hosts, tb_hosts, uuid_reverse_index)
          return
        end

        ips = IPConfigurator.new(hosts, tb_hosts, uuid_reverse_index)
        connectivity = make_connectivity_matrix(tb_hosts, uuid_reverse_index)
        backend = cfg_arg(args, :backend, nil)

        if backend.nil?
          rings = extract_rings(connectivity)
          has_mesh = check_valid_mesh(hosts, connectivity, strict: false)
          has_ring = check_valid_ring(hosts, rings, strict: false)
          has_rdma = check_rdma(hosts, cfg_arg(args, :verbose, false), strict: false, runner: runner)

          if !has_ring && !has_mesh
            raise ArgumentError, "Neither thunderbolt mesh nor ring found."
          elsif has_rdma && has_mesh
            configure_jaccl(args, hosts, ips, sshinfo, runner: runner)
          elsif has_rdma && has_ring
            configure_jaccl_ring(args, hosts, ips, rings[0], sshinfo, runner: runner)
          elsif has_ring
            configure_ring(args, hosts, ips, rings[0], sshinfo, runner: runner)
          else
            raise ArgumentError, "RDMA is not available and ring is not found."
          end
        elsif backend == "ring"
          rings = extract_rings(connectivity)
          check_valid_ring(hosts, rings)
          configure_ring(args, hosts, ips, rings[0], sshinfo, runner: runner)
        elsif backend == "jaccl"
          check_valid_mesh(hosts, connectivity)
          check_rdma(hosts, cfg_arg(args, :verbose, false), runner: runner)
          configure_jaccl(args, hosts, ips, sshinfo, runner: runner)
        elsif backend == "jaccl-ring"
          rings = extract_rings(connectivity)
          check_valid_ring(hosts, rings)
          check_rdma(hosts, cfg_arg(args, :verbose, false), runner: runner)
          configure_jaccl_ring(args, hosts, ips, rings[0], sshinfo, runner: runner)
        end
      end

      def config_main(argv = ARGV, runner: nil)
        opts = {
          verbose: false,
          hosts: "127.0.0.1",
          hostfile: nil,
          over: "thunderbolt",
          output_hostfile: nil,
          auto_setup: nil,
          dot: false,
          backend: nil,
          env: []
        }

        parser = OptionParser.new do |o|
          o.on("--verbose") { opts[:verbose] = true }
          o.on("--hosts HOSTS") { |v| opts[:hosts] = v }
          o.on("--hostfile FILE") { |v| opts[:hostfile] = v }
          o.on("--over TYPE") { |v| opts[:over] = v }
          o.on("--output-hostfile FILE") { |v| opts[:output_hostfile] = v }
          o.on("--auto-setup") { opts[:auto_setup] = true }
          o.on("--no-auto-setup") { opts[:auto_setup] = false }
          o.on("--dot") { opts[:dot] = true }
          o.on("--backend BACKEND") { |v| opts[:backend] = v }
          o.on("--env ENV") { |v| opts[:env] << v }
        end
        parser.parse!(argv.dup)

        hosts = if opts[:hostfile]
          Hostfile.from_file(opts[:hostfile]).hosts
        else
          Hostfile.from_list(opts[:hosts]).hosts
        end

        log(opts[:verbose], "Checking for ssh access for #{hosts.map(&:ssh_hostname).join(', ')}")
        sshinfo = check_ssh_connections(hosts, runner: runner)

        args = opts
        if opts[:over] == "ethernet"
          prepare_ethernet_hostfile(args, hosts, runner: runner)
        else
          prepare_tb_hostfile(args, hosts, sshinfo, runner: runner)
        end
        0
      end
      private :config_main

      private

      def cfg_arg(args, key, default)
        if args.respond_to?(key)
          value = args.public_send(key)
          value.nil? ? default : value
        elsif args.respond_to?(:[])
          value = args[key]
          value.nil? ? default : value
        else
          default
        end
      end

      def read_ip(hostname, iface, runner: nil)
        result = run_command(["ssh", hostname, "ipconfig", "getifaddr", iface], runner: runner)
        stdout_for(result).strip
      end

      def run_command(cmd, runner: nil)
        return runner.call(cmd) unless runner.nil?

        stdout, stderr, status = Open3.capture3(*cmd)
        Struct.new(:stdout, :stderr, :status, keyword_init: true).new(stdout: stdout, stderr: stderr, status: status)
      end

      def stdout_for(result)
        if result.respond_to?(:stdout)
          result.stdout.to_s
        else
          ""
        end
      end

      def success_for(result)
        status = result.respond_to?(:status) ? result.status : result
        return status.success? if status.respond_to?(:success?)
        return status.exitstatus.zero? if status.respond_to?(:exitstatus)

        true
      end
    end
  end
end
