# frozen_string_literal: true

require "pathname"
require "rbconfig"
require "stringio"
require "tempfile"
require "timeout"
require "fileutils"
require_relative "../test_helper"

class DocExamplesTest < Minitest::Test
  BLOCK_MARKERS = [
    /^\s*\.\.\s*code(?:-block)?::\s*ruby\s*$/,
    /^\s*\.\.\s*code::\s*ruby\s*$/
  ].freeze

  CHILD_TIMEOUT_SECONDS = (ENV["DOCS_TEST_CHILD_TIMEOUT"] || "180").to_i

  Block = Struct.new(:index, :header_line, :code_start_line, :code, keyword_init: true)
  ChildResult = Struct.new(:path, :success, :message, :output, keyword_init: true)

  # Docs validation spans many files and runs each one in a child process.
  # Use the original Minitest run implementation without the global 10s timeout.
  def run
    run_without_timeout
  end

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_ruby_doc_examples
    if child_mode?
      failures = run_examples_in_process
      if failures.empty?
        Process.exit!(0)
      else
        $stderr.puts(format_block_failures(failures))
        Process.exit!(1)
      end
    end

    failures = []

    rst_files.each do |rst_path|
      result = run_file_in_child(rst_path)
      failures << result unless result.success
    end

    assert failures.empty?, format_child_failures(failures)
  end

  private

  def child_mode?
    ENV["DOCS_TEST_CHILD"] == "1"
  end

  def run_examples_in_process
    failures = []

    rst_files.each do |rst_path|
      blocks = extract_ruby_blocks(rst_path)
      next if blocks.empty?

      context = build_context
      workdir = docs_test_workdir_for(rst_path)
      FileUtils.rm_rf(workdir)
      FileUtils.mkdir_p(workdir)

      Dir.chdir(workdir) do
        blocks.each do |block|
          max_block = ENV["DOCS_TEST_MAX_BLOCK"]&.to_i
          if max_block && max_block.positive? && block.index > max_block
            next
          end

          mode = block_mode(block.code)
          next if mode[:skip]

          trace_block(rst_path, block)

          begin
            runner = lambda do
              if mode[:timeout_seconds].nil?
                eval(block.code, context, rst_path, block.code_start_line)
              else
                Timeout.timeout(mode[:timeout_seconds]) do
                  eval(block.code, context, rst_path, block.code_start_line)
                end
              end
            end

            if mode[:skip_capture]
              runner.call
            else
              capture_output { runner.call }
            end

            if mode[:expect_error]
              failures << failure_hash(rst_path, block, "expected error but block succeeded")
            end
          rescue Exception => e # rubocop:disable Lint/RescueException
            unless mode[:expect_error]
              failures << failure_hash(rst_path, block, "#{e.class}: #{e.message.lines.first}")
            end
          end
        end
      end
    end

    failures
  end

  def docs_test_workdir_for(rst_path)
    relative_path = Pathname.new(rst_path).relative_path_from(Pathname.new(RUBY_ROOT)).to_s
    slug = relative_path.gsub(File::SEPARATOR, "__").gsub(/[^A-Za-z0-9_.-]/, "_")
    File.join(RUBY_ROOT, "tmp", "docs-test-artifacts", slug)
  end

  def run_file_in_child(rst_path)
    relative_path = Pathname.new(rst_path).relative_path_from(Pathname.new(RUBY_ROOT)).to_s
    stdout_file = Tempfile.new("docs-test-out")
    stderr_file = Tempfile.new("docs-test-err")

    env = {
      "DOCS_TEST_CHILD" => "1",
      "DOCS_TEST_FILE" => relative_path,
      "DOCS_TEST_MAX_BLOCK" => ENV["DOCS_TEST_MAX_BLOCK"].to_s,
      "DOCS_TEST_TRACE" => ENV["DOCS_TEST_TRACE"].to_s
    }

    pid = Process.spawn(
      env,
      RbConfig.ruby,
      "-Itest",
      __FILE__,
      chdir: RUBY_ROOT,
      out: stdout_file.path,
      err: stderr_file.path
    )

    timed_out, process_status = wait_for_child(pid, CHILD_TIMEOUT_SECONDS)

    stdout_file.rewind
    stderr_file.rewind
    stdout = stdout_file.read
    stderr = stderr_file.read
    stdout_file.close!
    stderr_file.close!

    if timed_out
      begin
        Process.kill("KILL", pid)
      rescue Errno::ESRCH
        nil
      end

      begin
        Process.wait(pid)
      rescue Errno::ECHILD
        nil
      end

      output = [stdout, stderr].join
      return ChildResult.new(
        path: rst_path,
        success: false,
        message: "timed out after #{CHILD_TIMEOUT_SECONDS}s",
        output: output
      )
    end

    success = process_status&.success? == true
    ChildResult.new(
      path: rst_path,
      success: success,
      message: success ? "" : child_status_message(process_status),
      output: [stdout, stderr].join
    )
  end

  def wait_for_child(pid, timeout_seconds)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout_seconds

    loop do
      waited_pid, status = Process.waitpid2(pid, Process::WNOHANG)
      return [false, status] if waited_pid

      return [true, nil] if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline

      sleep 0.1
    end
  end

  def child_status_message(process_status)
    return "child exited without status" if process_status.nil?
    return "child exited with status #{process_status.exitstatus}" if process_status.exited?
    return "child terminated by signal #{process_status.termsig}" if process_status.signaled?

    "child exited abnormally (status=#{process_status.inspect})"
  end

  def rst_files
    @rst_files ||= begin
      if ENV["DOCS_TEST_FILE"] && !ENV["DOCS_TEST_FILE"].strip.empty?
        [File.expand_path("../../#{ENV["DOCS_TEST_FILE"]}", __dir__)]
      else
        list_path = File.join(__dir__, "rst_files.txt")
        File.readlines(list_path, chomp: true).reject(&:empty?).map do |path|
          File.expand_path("../../#{path}", __dir__)
        end
      end
    end
  end

  def extract_ruby_blocks(path)
    lines = File.readlines(path)
    blocks = []
    index = 0
    i = 0

    while i < lines.length
      line = lines[i]
      unless BLOCK_MARKERS.any? { |marker| line.match?(marker) }
        i += 1
        next
      end

      index += 1
      header_line = i + 1
      i += 1
      i += 1 while i < lines.length && lines[i].strip.empty?

      indent = nil
      code_lines = []
      while i < lines.length
        current = lines[i]
        if current.strip.empty?
          code_lines << current
          i += 1
          next
        end

        current_indent = current[/^\s*/].size
        indent ||= current_indent
        break if current_indent < indent

        code_lines << current[indent..]
        i += 1
      end

      blocks << Block.new(
        index: index,
        header_line: header_line,
        code_start_line: header_line + 2,
        code: code_lines.join
      )
    end

    blocks
  end

  def build_context
    context_obj = Object.new
    context_obj.define_singleton_method(:mx) { MLX::Core }
    context_obj.define_singleton_method(:nn) { MLX::NN }
    context_obj.define_singleton_method(:optim) { MLX::Optimizers }
    context_obj.instance_eval { binding }
  end

  def block_mode(code)
    mode = {
      expect_error: false,
      skip: false,
      skip_capture: false,
      timeout_seconds: 10
    }

    code.each_line do |line|
      stripped = line.strip
      mode[:expect_error] = true if stripped == "# docs-test: expect-error"
      mode[:skip] = true if stripped.start_with?("# docs-test: skip")
      if stripped.start_with?("# docs-test: timeout=")
        timeout = stripped.split("=", 2).last.to_i
        mode[:timeout_seconds] = timeout if timeout.positive?
      end
      mode[:timeout_seconds] = nil if stripped == "# docs-test: no-timeout"
      mode[:skip_capture] = true if stripped == "# docs-test: no-capture"
    end

    # Running compile snippets with redirected stdout/stderr can be unstable.
    #
    # Also avoid Timeout.timeout around compile examples; on some Ruby/Linux
    # CI combinations this can segfault inside timeout thread cleanup when a
    # native extension is active.
    if code.include?("mx.compile(") || code.include?("MLX::Core.compile(")
      mode[:skip_capture] = true
      mode[:timeout_seconds] = nil
    end

    mode
  end

  def capture_output
    original_stdout = $stdout
    original_stderr = $stderr
    $stdout = StringIO.new
    $stderr = StringIO.new
    yield
  ensure
    $stdout = original_stdout
    $stderr = original_stderr
  end

  def failure_hash(path, block, message)
    {
      path: path,
      block: block.index,
      line: block.code_start_line,
      message: message.strip,
      first_line: block.code.each_line.find { |line| !line.strip.empty? }&.strip
    }
  end

  def trace_block(rst_path, block)
    return unless ENV["DOCS_TEST_TRACE"] == "1"

    $stderr.puts("DOCS_TEST_TRACE #{File.basename(rst_path)} block=#{block.index} line=#{block.code_start_line}")
  end

  def format_block_failures(failures)
    lines = []
    lines << "#{failures.length} doc example failure(s):"
    failures.each do |failure|
      lines << "- #{failure[:path]}:#{failure[:line]} block #{failure[:block]}"
      lines << "  #{failure[:message]}"
      lines << "  first line: #{failure[:first_line]}"
    end
    lines.join("\n")
  end

  def format_child_failures(failures)
    lines = []
    lines << "#{failures.length} doc file failure(s):"

    failures.each do |failure|
      lines << "- #{failure.path}"
      lines << "  #{failure.message}"

      output_lines = failure.output.to_s.lines.map(&:chomp)
      next if output_lines.empty?

      lines << "  child output (last 40 lines):"
      output_lines.last(40).each do |line|
        lines << "    #{line}"
      end
    end

    lines.join("\n")
  end
end
