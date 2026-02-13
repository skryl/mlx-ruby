# frozen_string_literal: true

require "ipaddr"
require "json"

module MLX
  module DistributedUtils
    Host = Struct.new(:rank, :ssh_hostname, :ips, :rdma, keyword_init: true)

    class Hostfile
      attr_accessor :hosts, :backend, :envs

      def initialize(hosts, backend = "", envs = [])
        @hosts = hosts
        @backend = backend
        @envs = envs
      end

      def to_json_obj
        {
          "backend" => backend,
          "envs" => envs,
          "hosts" => hosts.map do |h|
            { "ssh" => h.ssh_hostname, "ips" => h.ips, "rdma" => h.rdma }
          end
        }
      end

      def self.from_file(hostfile)
        unless File.exist?(hostfile)
          raise ArgumentError, "Hostfile #{hostfile} doesn't exist"
        end

        begin
          data = JSON.parse(File.read(hostfile))
          backend = ""
          envs = []
          hosts_data = []
          if data.is_a?(Hash)
            backend = data.fetch("backend", "")
            envs = data.fetch("envs", [])
            hosts_data = data.fetch("hosts", [])
          elsif data.is_a?(Array)
            hosts_data = data
          end

          hosts = hosts_data.each_with_index.map do |h, i|
            Host.new(
              rank: i,
              ssh_hostname: h.fetch("ssh"),
              ips: h.fetch("ips", []),
              rdma: h.fetch("rdma", [])
            )
          end
          new(hosts, backend, envs)
        rescue StandardError => e
          raise ArgumentError, "Failed to parse hostfile #{hostfile} (#{e})"
        end
      end

      def self.from_list(hostlist, repeats = 1)
        hosts = []
        rank = 0
        hostlist.split(",").each do |h|
          raise ArgumentError, "Hostname cannot be empty" if h.empty?

          ips = []
          begin
            IPAddr.new(h)
            ips = [h]
          rescue IPAddr::InvalidAddressError
          end

          repeats.times do
            hosts << Host.new(rank: rank, ssh_hostname: h, ips: ips.dup, rdma: [])
            rank += 1
          end
        end
        new(hosts)
      end
    end

    class OptionalBoolAction
      def call(_parser, namespace, _values, option_string = nil)
        key = option_string.to_s.sub(/\A--no-/, "").sub(/\A--/, "").tr("-", "_").to_sym
        namespace[key] = !option_string.to_s.start_with?("--no-")
      end
    end

    class << self
      def positive_number(value)
        number = Integer(value)
        raise ArgumentError, "Number should be positive" unless number.positive?

        number
      end

      def log(verbose, *args, **kwargs)
        return unless verbose

        _ = kwargs
        $stderr.puts("\e[32m[INFO] #{args.join(' ')} \e[0m")
      end

      def log_warning(*args, **kwargs)
        _ = kwargs
        $stderr.puts("\e[33m[WARN] #{args.join(' ')} \e[0m")
      end

      def log_error(*args, **kwargs)
        _ = kwargs
        $stderr.puts("\e[31m[ERROR] #{args.join(' ')} \e[0m")
      end
    end
  end
end
