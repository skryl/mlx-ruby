import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";

const STATUS = document.getElementById("status");
const INPUTS_TEXTAREA = document.getElementById("inputs-json");
const RUN_BUTTON = document.getElementById("run-button");
const SUPPORTED_PROVIDERS = ["webgpu", "wasm"];

function setStatus(value) {
  STATUS.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function flattenValues(value, out = []) {
  if (Array.isArray(value)) {
    for (const item of value) {
      flattenValues(item, out);
    }
    return out;
  }
  out.push(value);
  return out;
}

function tensorTypeForDtype(dtype) {
  switch (dtype) {
    case "bool":
    case "bool_":
      return "bool";
    case "uint8":
      return "uint8";
    case "int8":
      return "int8";
    case "uint16":
      return "uint16";
    case "int16":
      return "int16";
    case "uint32":
      return "uint32";
    case "int32":
      return "int32";
    case "uint64":
      return "uint64";
    case "int64":
      return "int64";
    case "float16":
      return "float32";
    case "float32":
      return "float32";
    case "bfloat16":
      return "float32";
    case "float64":
      return "float64";
    default:
      throw new Error(`unsupported input dtype in harness: ${dtype}`);
  }
}

function tensorDataForType(type, values) {
  switch (type) {
    case "bool":
      return Uint8Array.from(values.map((v) => (v ? 1 : 0)));
    case "uint8":
      return Uint8Array.from(values.map((v) => Number(v)));
    case "int8":
      return Int8Array.from(values.map((v) => Number(v)));
    case "uint16":
      return Uint16Array.from(values.map((v) => Number(v)));
    case "int16":
      return Int16Array.from(values.map((v) => Number(v)));
    case "uint32":
      return Uint32Array.from(values.map((v) => Number(v)));
    case "int32":
      return Int32Array.from(values.map((v) => Number(v)));
    case "uint64":
      return BigUint64Array.from(values.map((v) => BigInt(v)));
    case "int64":
      return BigInt64Array.from(values.map((v) => BigInt(v)));
    case "float64":
      return Float64Array.from(values.map((v) => Number(v)));
    case "float32":
      return Float32Array.from(values.map((v) => Number(v)));
    default:
      throw new Error(`unsupported tensor type in harness: ${type}`);
  }
}

function buildTensor(spec, rawValue) {
  const dtype = spec.dtype;
  if (dtype === "complex64") {
    throw new Error("complex64 inputs are not currently supported in the web harness");
  }
  const type = tensorTypeForDtype(dtype);
  const values = flattenValues(rawValue);
  const data = tensorDataForType(type, values);
  return new ort.Tensor(type, data, spec.shape);
}

function normalizeJsonScalar(value) {
  if (typeof value === "bigint") {
    return Number(value);
  }
  return value;
}

function reshapeFlatValues(flatValues, shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return flatValues[0] ?? null;
  }

  const out = [];
  if (shape.length === 1) {
    for (let i = 0; i < shape[0]; i += 1) {
      out.push(flatValues[i]);
    }
    return out;
  }

  const chunkSize = shape.slice(1).reduce((acc, v) => acc * v, 1);
  for (let i = 0; i < shape[0]; i += 1) {
    const start = i * chunkSize;
    const finish = start + chunkSize;
    out.push(reshapeFlatValues(flatValues.slice(start, finish), shape.slice(1)));
  }
  return out;
}

function serializeTensorOutput(value) {
  if (!value || !Array.isArray(value.dims) || value.data == null) {
    return value;
  }
  const flat = Array.from(value.data, normalizeJsonScalar);
  return reshapeFlatValues(flat, value.dims);
}

function serializeRunOutputs(outputs) {
  const serialized = {};
  for (const [name, value] of Object.entries(outputs)) {
    serialized[name] = serializeTensorOutput(value);
  }
  return serialized;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`failed to fetch ${path}: HTTP ${response.status}`);
  }
  return response.json();
}

async function createSessionWithFallback(modelPath, providers) {
  const errors = [];
  for (const provider of providers) {
    if (!SUPPORTED_PROVIDERS.includes(provider)) {
      throw new Error(`manifest contains unsupported provider ${provider}`);
    }
  }
  for (let i = 0; i < providers.length; i += 1) {
    const provider = providers[i];
    try {
      const session = await ort.InferenceSession.create(modelPath, {
        executionProviders: [provider]
      });
      return { session, provider, fallbackIndex: i, errors };
    } catch (error) {
      errors.push({
        provider,
        message: String(error)
      });
    }
  }
  throw new Error(`failed to initialize ONNX Runtime session for providers ${providers.join(", ")}`);
}

async function runBenchmark(session, feeds, warmupRuns, measureRuns) {
  for (let i = 0; i < warmupRuns; i += 1) {
    await session.run(feeds);
  }

  const timings = [];
  let sampleOutputs = null;
  for (let i = 0; i < measureRuns; i += 1) {
    const start = performance.now();
    const outputs = await session.run(feeds);
    if (sampleOutputs === null) {
      sampleOutputs = serializeRunOutputs(outputs);
    }
    timings.push(performance.now() - start);
  }

  const total = timings.reduce((sum, v) => sum + v, 0);
  return {
    runs_ms: timings,
    first_inference_ms: timings[0] ?? 0,
    steady_state_inference_ms: timings.length > 0 ? total / timings.length : 0,
    sample_outputs: sampleOutputs || {}
  };
}

function buildFeeds(inputSpecs, inputPayload) {
  const feeds = {};
  for (const spec of inputSpecs) {
    const name = spec.name;
    if (!(name in inputPayload)) {
      throw new Error(`missing input payload for ${name}`);
    }
    feeds[name] = buildTensor(spec, inputPayload[name]);
  }
  return feeds;
}

async function loadDefaultInputs() {
  const sampleInputs = await fetchJson("./inputs.example.json");
  INPUTS_TEXTAREA.value = JSON.stringify(sampleInputs, null, 2);
}

async function run() {
  RUN_BUTTON.disabled = true;
  try {
    setStatus("Loading harness manifest...");
    const manifest = await fetchJson("./harness.manifest.json");
    const modelPath = `./${manifest.model}`;
    const providers = manifest.execution_providers;

    setStatus("Initializing session...");
    const sessionStart = performance.now();
    const sessionResult = await createSessionWithFallback(modelPath, providers);
    const sessionCreateMs = performance.now() - sessionStart;

    const parsedInputPayload = JSON.parse(INPUTS_TEXTAREA.value);
    const feeds = buildFeeds(manifest.inputs, parsedInputPayload);
    const benchmark = await runBenchmark(
      sessionResult.session,
      feeds,
      manifest.benchmark.warmup_runs,
      manifest.benchmark.measure_runs
    );

    const telemetry = {
      format: "onnx_webgpu_telemetry_v1",
      selected_provider: sessionResult.provider,
      requested_providers: providers,
      fallback_used: sessionResult.fallbackIndex > 0,
      fallback_partition_ratio: sessionResult.fallbackIndex > 0 ? 1.0 : 0.0,
      model_compile_latency_ms: sessionCreateMs,
      model_load_latency_ms: sessionCreateMs,
      first_inference_latency_ms: benchmark.first_inference_ms,
      steady_state_inference_latency_ms: benchmark.steady_state_inference_ms,
      run_timings_ms: benchmark.runs_ms,
      sample_outputs: benchmark.sample_outputs,
      provider_init_errors: sessionResult.errors
    };
    setStatus(telemetry);
  } catch (error) {
    setStatus({ error: String(error) });
  } finally {
    RUN_BUTTON.disabled = false;
  }
}

RUN_BUTTON.addEventListener("click", run);
loadDefaultInputs()
  .then(() => setStatus("Ready. Click 'Run Benchmark' to execute."))
  .catch((error) => setStatus({ error: String(error) }));
