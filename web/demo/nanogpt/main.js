const ASSET_ROOT_CANDIDATES = Array.from(
  new Set([
    new URL("../../assets/nanogpt", import.meta.url).toString().replace(/\/$/, ""),
    new URL("../assets/nanogpt", import.meta.url).toString().replace(/\/$/, ""),
    new URL("./assets/nanogpt", import.meta.url).toString().replace(/\/$/, "")
  ])
);
let ASSET_ROOT = ASSET_ROOT_CANDIDATES[0];
let MODEL_PATH = "";
let META_PATH = "";
let PRESETS_PATH = "";
let TOKENIZER_PATH = "";
const ORT_MODULE_CANDIDATES = [
  "../../node_modules/onnxruntime-web/dist/ort.all.min.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs"
];

setAssetRoot(ASSET_ROOT);

function setAssetRoot(root) {
  ASSET_ROOT = root;
  MODEL_PATH = `${ASSET_ROOT}/model.onnx`;
  META_PATH = `${ASSET_ROOT}/meta.json`;
  PRESETS_PATH = `${ASSET_ROOT}/prompt.presets.json`;
  TOKENIZER_PATH = `${ASSET_ROOT}/tokenizer.json`;
}

const weightsBadge = document.getElementById("badge-weights");
const providerBadge = document.getElementById("badge-provider");
const onnxSizeBadge = document.getElementById("badge-onnx-size");
const parametersBadge = document.getElementById("badge-parameters");
const timingBadge = document.getElementById("badge-timing");
const modelStatus = document.getElementById("status-model");
const shapeStatus = document.getElementById("status-shape");
const tokenizerStatus = document.getElementById("status-tokenizer");
const contextStatus = document.getElementById("status-context");
const onnxSizeStatus = document.getElementById("status-onnx-size");
const errorBox = document.getElementById("error-box");

const presetSelect = document.getElementById("preset");
const presetName = document.getElementById("preset-name");
const promptInput = document.getElementById("prompt");
const temperatureInput = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const maxTokensInput = document.getElementById("max-tokens");
const maxTokensValue = document.getElementById("max-tokens-value");
const generateButton = document.getElementById("generate");
const clearButton = document.getElementById("clear");

const outputText = document.getElementById("output-text");
const outputIds = document.getElementById("output-ids");
const topKBody = document.getElementById("topk-body");

let session = null;
let selectedProvider = null;
let modelMeta = null;
let promptPresets = {};
let tokenizerConfig = null;
let running = false;
let ort = null;

generateButton.disabled = true;
setGenerationEnabled(false);

function showError(message) {
  errorBox.style.display = "block";
  errorBox.textContent = message;
}

function clearError() {
  errorBox.style.display = "none";
  errorBox.textContent = "";
}

function missingAssetsMessage() {
  const probes = ASSET_ROOT_CANDIDATES.map((candidate) => `${candidate}/meta.json`);
  return [
    "Demo assets were not found on this host.",
    "GitHub Pages deployments often omit generated web/assets checkpoints.",
    "Checked:",
    ...probes.map((probe) => `- ${probe}`),
    "Run `bundle exec rake web:assets` and serve with `bundle exec rake web:serve`."
  ].join("\n");
}

function toNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function setGenerationEnabled(enabled) {
  generateButton.disabled = !enabled;
}

function setOnnxSizeText(text) {
  onnxSizeStatus.textContent = text;
  if (onnxSizeBadge) {
    onnxSizeBadge.textContent = text;
  }
}

function formatParameterCount(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "unknown";
  }
  return Math.trunc(numeric).toLocaleString("en-US");
}

function setParameterText(value) {
  if (!parametersBadge) {
    return;
  }
  parametersBadge.textContent = `Parameters: ${formatParameterCount(value)}`;
}

function formatByteSize(bytes) {
  const numeric = Number(bytes);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "unknown";
  }
  if (numeric < 1024) {
    return `${Math.trunc(numeric)} B`;
  }

  const units = ["KB", "MB", "GB", "TB"];
  let value = numeric;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const precision = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(precision)} ${units[unitIndex]}`;
}

async function onnxSizeBytes(path) {
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    const contentLength = response.headers.get("content-length");
    const parsed = Number.parseInt(contentLength || "", 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
      return null;
    }
    return parsed;
  } catch (_error) {
    return null;
  }
}

async function updateOnnxSizeStatus() {
  const bytes = await onnxSizeBytes(MODEL_PATH);
  if (bytes === null) {
    setOnnxSizeText("ONNX Size: unavailable");
    return;
  }
  setOnnxSizeText(`ONNX Size: ${formatByteSize(bytes)}`);
}

async function assetExists(path) {
  try {
    const response = await fetch(path, { method: "HEAD", cache: "no-store" });
    if (response.ok) {
      return true;
    }
    if (response.status === 405 || response.status === 501) {
      const fallback = await fetch(path, { cache: "no-store" });
      return fallback.ok;
    }
    return false;
  } catch (_error) {
    return false;
  }
}

async function resolveAssetRoot() {
  for (const candidate of ASSET_ROOT_CANDIDATES) {
    if (await assetExists(`${candidate}/meta.json`)) {
      setAssetRoot(candidate);
      return true;
    }
  }
  return false;
}

function hasUsableWeights(meta) {
  return meta?.trained === true;
}

function setMissingWeightsState(message) {
  const detail = message ? ` (${message})` : "";
  weightsBadge.textContent = `Weights: missing${detail}`;
  providerBadge.textContent = "Provider: unavailable";
  modelStatus.textContent = "Model: unavailable (missing weights)";
  setGenerationEnabled(false);
}

function tokenizer() {
  return tokenizerConfig ?? modelMeta?.tokenizer ?? {};
}

function loadJson(path) {
  return fetch(path, { cache: "no-store" }).then((response) => {
    if (!response.ok) {
      throw new Error(`failed to fetch ${path}: HTTP ${response.status}`);
    }
    return response.json();
  });
}

async function loadOrtModule() {
  const errors = [];
  for (const source of ORT_MODULE_CANDIDATES) {
    try {
      const mod = await import(source);
      if (mod && mod.InferenceSession && mod.Tensor) {
        return mod;
      }
      errors.push(`${source}: missing expected exports`);
    } catch (error) {
      errors.push(`${source}: ${String(error)}`);
    }
  }
  throw new Error(
    `failed to load ONNX Runtime Web module:\n${errors.map((entry) => `- ${entry}`).join("\n")}`
  );
}

async function createSessionWithFallback() {
  const providers = ["webgpu", "wasm"];
  const errors = [];
  for (const provider of providers) {
    try {
      const created = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: [provider]
      });
      return { session: created, provider, errors };
    } catch (error) {
      errors.push({ provider, message: String(error) });
    }
  }
  throw new Error(
    `failed to initialize ONNX Runtime session for providers ${providers.join(", ")}:\n` +
      `${errors.map((entry) => `- ${entry.provider}: ${entry.message}`).join("\n")}`
  );
}

function promptToIds(prompt) {
  const cfg = tokenizer();
  const padId = toNumber(cfg.pad_id, 0);
  const unkId = Number.isFinite(Number(cfg.unk_id)) ? toNumber(cfg.unk_id, padId) : padId;
  const charToId = cfg.char_to_id ?? {};
  const ids = [];
  if (Number.isFinite(Number(cfg.bos_id))) {
    ids.push(toNumber(cfg.bos_id, padId));
  }
  const chars = String(prompt || "").split("");
  chars.forEach((char) => {
    ids.push(toNumber(charToId[char], unkId));
  });
  return ids.length > 0 ? ids : [padId];
}

function idsToText(ids) {
  const cfg = tokenizer();
  const idToChar = cfg.id_to_char ?? {};
  const padId = Number.isFinite(Number(cfg.pad_id)) ? toNumber(cfg.pad_id, 0) : null;
  const bosId = Number.isFinite(Number(cfg.bos_id)) ? toNumber(cfg.bos_id, 0) : null;
  const eosId = Number.isFinite(Number(cfg.eos_id)) ? toNumber(cfg.eos_id, 0) : null;
  return ids.map((id) => {
    const key = String(id);
    if (key in idToChar) {
      return idToChar[key];
    }
    if ((padId !== null && id === padId) || (bosId !== null && id === bosId) || (eosId !== null && id === eosId)) {
      return "";
    }
    return "�";
  }).join("");
}

function tokenLabel(id) {
  const cfg = tokenizer();
  const idToChar = cfg.id_to_char ?? {};
  if (String(id) in idToChar) {
    return JSON.stringify(idToChar[String(id)]);
  }
  if (Number.isFinite(Number(cfg.pad_id)) && id === toNumber(cfg.pad_id, 0)) return "<pad>";
  if (Number.isFinite(Number(cfg.bos_id)) && id === toNumber(cfg.bos_id, 0)) return "<bos>";
  if (Number.isFinite(Number(cfg.eos_id)) && id === toNumber(cfg.eos_id, 0)) return "<eos>";
  return "<unk>";
}

function contextSize() {
  const fromMeta = toNumber(modelMeta?.generation?.context_size, 0);
  if (fromMeta > 0) return fromMeta;
  return toNumber(modelMeta?.input?.shape?.[1], 12);
}

function buildContext(promptIds) {
  const size = contextSize();
  const padId = toNumber(tokenizer().pad_id, 0);
  const truncated = promptIds.slice(-size);
  if (truncated.length < size) {
    return Array(size - truncated.length).fill(padId).concat(truncated);
  }
  return truncated;
}

function normalizeInputShape(shape, tokenCount) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return [1, tokenCount];
  }

  return shape.map((entry, index) => {
    const value = Number(entry);
    if (Number.isFinite(value) && value > 0) {
      return Math.trunc(value);
    }
    if (index === shape.length - 1) {
      return tokenCount;
    }
    return 1;
  });
}

function inputTensorType(rawType) {
  const normalized = String(rawType || "").toLowerCase();
  return normalized.includes("64") ? "int64" : "int32";
}

function createFeeds(tokens) {
  const inputName = (session?.inputNames || [])[0] || modelMeta?.input?.name || "input";
  const sessionInputMeta = session?.inputMetadata?.[inputName] || {};
  const inputShape = normalizeInputShape(
    sessionInputMeta.dimensions || modelMeta?.input?.shape,
    tokens.length
  );
  const tensorType = inputTensorType(sessionInputMeta.type || modelMeta?.input?.type);

  if (tensorType === "int64") {
    return {
      [inputName]: new ort.Tensor(
        "int64",
        BigInt64Array.from(tokens.map((value) => BigInt(value))),
        inputShape
      )
    };
  }

  return {
    [inputName]: new ort.Tensor("int32", Int32Array.from(tokens), inputShape)
  };
}

function outputTensor(outputs) {
  if (modelMeta.output.name in outputs) {
    return outputs[modelMeta.output.name];
  }
  const values = Object.values(outputs);
  if (values.length === 0) {
    throw new Error("runtime returned no outputs");
  }
  return values[0];
}

function lastStepLogits(tensor) {
  const dims = tensor.dims || [];
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`unexpected output shape: ${JSON.stringify(dims)}`);
  }
  const seqLen = dims[1];
  const vocabSize = dims[2];
  const start = (seqLen - 1) * vocabSize;
  const end = start + vocabSize;
  return tensor.data.slice(start, end);
}

function argmax(values) {
  let bestIdx = 0;
  let bestValue = Number(values[0]);
  for (let i = 1; i < values.length; i += 1) {
    const value = Number(values[i]);
    if (value > bestValue) {
      bestValue = value;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function sampleFromLogits(values, temperature) {
  if (temperature <= 0.0001) {
    return argmax(values);
  }

  let maxValue = -Infinity;
  const scaled = new Float64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const value = Number(values[i]) / temperature;
    scaled[i] = value;
    if (value > maxValue) {
      maxValue = value;
    }
  }

  let sum = 0.0;
  for (let i = 0; i < scaled.length; i += 1) {
    const expValue = Math.exp(scaled[i] - maxValue);
    scaled[i] = expValue;
    sum += expValue;
  }

  let threshold = Math.random() * sum;
  for (let i = 0; i < scaled.length; i += 1) {
    threshold -= scaled[i];
    if (threshold <= 0) {
      return i;
    }
  }
  return scaled.length - 1;
}

function topK(values, k = 8) {
  let maxValue = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const value = Number(values[i]);
    if (value > maxValue) {
      maxValue = value;
    }
  }

  let normalizer = 0.0;
  const expValues = new Float64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const expValue = Math.exp(Number(values[i]) - maxValue);
    expValues[i] = expValue;
    normalizer += expValue;
  }

  const entries = Array.from({ length: values.length }, (_, id) => ({
    id,
    logit: Number(values[id]),
    prob: expValues[id] / normalizer
  }));

  entries.sort((a, b) => b.logit - a.logit);
  return entries.slice(0, k);
}

function renderTopK(values) {
  const rows = topK(values, 8);
  topKBody.innerHTML = "";
  rows.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>${tokenLabel(entry.id)}</td>
      <td>${(entry.prob * 100).toFixed(2)}%</td>
      <td>${entry.logit.toFixed(4)}</td>
    `;
    topKBody.appendChild(row);
  });
}

function setTemperatureLabel() {
  temperatureValue.textContent = toNumber(temperatureInput.value, 0.9).toFixed(2);
}

function setMaxTokensLabel() {
  maxTokensValue.textContent = String(Math.max(1, Math.min(256, toNumber(maxTokensInput.value, 80))));
}

function installPresets() {
  presetSelect.innerHTML = "";
  Object.keys(promptPresets).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    presetSelect.appendChild(option);
  });
}

function applyPreset(name) {
  if (!(name in promptPresets)) {
    return;
  }
  presetSelect.value = name;
  presetName.textContent = name;
  promptInput.value = promptPresets[name];
}

async function runGeneration() {
  if (!session || !modelMeta || running) {
    return;
  }

  running = true;
  setGenerationEnabled(false);
  clearError();

  try {
    const temperature = Math.max(0.0, toNumber(temperatureInput.value, 0.9));
    const maxTokens = Math.max(1, Math.min(256, toNumber(maxTokensInput.value, 80)));
    const eosId = Number.isFinite(Number(tokenizer().eos_id)) ? toNumber(tokenizer().eos_id, 0) : null;

    const promptIds = promptToIds(promptInput.value);
    let context = buildContext(promptIds);
    contextStatus.textContent = `Context tokens: ${context.join(" ")}`;

    const generatedIds = [];
    let firstStepLogits = null;
    let totalMs = 0.0;

    for (let step = 0; step < maxTokens; step += 1) {
      const started = performance.now();
      const outputs = await session.run(createFeeds(context));
      totalMs += performance.now() - started;

      const tensor = outputTensor(outputs);
      shapeStatus.textContent = `Output shape: ${JSON.stringify(tensor.dims)}`;
      const logits = lastStepLogits(tensor);
      if (firstStepLogits === null) {
        firstStepLogits = logits;
      }

      const nextToken = sampleFromLogits(logits, temperature);
      generatedIds.push(nextToken);
      context = context.slice(1).concat([nextToken]);

      if (eosId !== null && nextToken === eosId) {
        break;
      }
    }

    renderTopK(firstStepLogits ?? []);

    const renderedText = idsToText(generatedIds);
    outputText.textContent = renderedText.length === 0 ? "(no printable tokens)" : renderedText;
    outputIds.textContent = generatedIds.length === 0 ? "[]" : generatedIds.join(" ");

    const perToken = generatedIds.length > 0 ? totalMs / generatedIds.length : 0.0;
    timingBadge.textContent = `Inference: ${totalMs.toFixed(2)} ms total (${perToken.toFixed(2)} ms/token)`;
  } catch (error) {
    showError(String(error));
  } finally {
    running = false;
    setGenerationEnabled(true);
  }
}

function installUi() {
  presetSelect.addEventListener("change", () => {
    applyPreset(presetSelect.value);
  });

  temperatureInput.addEventListener("input", setTemperatureLabel);
  maxTokensInput.addEventListener("input", setMaxTokensLabel);

  generateButton.addEventListener("click", () => {
    runGeneration().catch((error) => showError(String(error)));
  });

  clearButton.addEventListener("click", () => {
    outputText.textContent = "";
    outputIds.textContent = "";
    topKBody.innerHTML = "";
    timingBadge.textContent = "Inference: -- ms";
  });
}

async function boot() {
  try {
    setGenerationEnabled(false);
    const assetRootResolved = await resolveAssetRoot();
    if (!assetRootResolved) {
      setOnnxSizeText("ONNX Size: unavailable");
      setParameterText(null);
      setMissingWeightsState("run `bundle exec rake web:assets`");
      modelStatus.textContent = "Model: unavailable (missing demo assets)";
      tokenizerStatus.textContent = "Tokenizer: unavailable";
      contextStatus.textContent = "Context size: unavailable";
      shapeStatus.textContent = "Output shape: unavailable";
      showError(missingAssetsMessage());
      return;
    }
    setOnnxSizeText("ONNX Size: probing...");
    const onnxSizeProbe = updateOnnxSizeStatus();
    modelStatus.textContent = "Runtime: loading ONNX Runtime Web...";
    ort = await loadOrtModule();
    modelStatus.textContent = "Model: loading metadata...";
    [modelMeta, promptPresets, tokenizerConfig] = await Promise.all([
      loadJson(META_PATH),
      loadJson(PRESETS_PATH),
      loadJson(TOKENIZER_PATH).catch(() => null)
    ]);
    await onnxSizeProbe;
    setParameterText(modelMeta?.parameters?.total ?? modelMeta?.parameter_count);
    if (!tokenizerConfig) {
      tokenizerConfig = modelMeta?.tokenizer ?? {};
    }

    installPresets();
    installUi();
    const defaultTemperature = toNumber(modelMeta?.generation?.default_temperature, 0.8);
    const defaultMaxTokens = toNumber(modelMeta?.generation?.default_max_tokens, 80);
    temperatureInput.value = String(defaultTemperature);
    maxTokensInput.value = String(defaultMaxTokens);
    setTemperatureLabel();
    setMaxTokensLabel();

    const firstPreset = Object.keys(promptPresets)[0];
    if (firstPreset) {
      applyPreset(firstPreset);
    }

    const hasModelAsset = await assetExists(MODEL_PATH);
    if (!hasUsableWeights(modelMeta) || !hasModelAsset) {
      setMissingWeightsState("run `bundle exec rake web:assets`");
      tokenizerStatus.textContent = "Tokenizer: unavailable";
      contextStatus.textContent = "Context size: unavailable";
      shapeStatus.textContent = "Output shape: unavailable";
      return;
    }

    modelStatus.textContent = "Model: creating runtime session...";
    const created = await createSessionWithFallback();
    session = created.session;
    selectedProvider = created.provider;
    providerBadge.textContent = `Provider: ${selectedProvider}`;
    if (selectedProvider !== "webgpu") {
      providerBadge.textContent += " (fallback)";
    }
    weightsBadge.textContent = "Weights: trained";

    const vocabSize = toNumber(tokenizer().vocab_size, 0);
    tokenizerStatus.textContent = `Tokenizer: ${tokenizer().type || "unknown"} (vocab=${vocabSize})`;
    modelStatus.textContent = `Model: ${modelMeta.model_name}`;
    contextStatus.textContent = `Context size: ${contextSize()} tokens`;
    shapeStatus.textContent = `Output shape: ${JSON.stringify(modelMeta.output.shape)}`;
    setGenerationEnabled(true);
  } catch (error) {
    showError(String(error));
    setMissingWeightsState("unavailable");
    setParameterText(null);
    tokenizerStatus.textContent = "Tokenizer: unavailable";
    contextStatus.textContent = "Context size: unavailable";
    shapeStatus.textContent = "Output shape: unavailable";
  }
}

boot();
