const ASSET_ROOT_CANDIDATES = Array.from(
  new Set([
    new URL("../../assets/gpt2", import.meta.url).toString().replace(/\/$/, ""),
    new URL("../assets/gpt2", import.meta.url).toString().replace(/\/$/, ""),
    new URL("./assets/gpt2", import.meta.url).toString().replace(/\/$/, "")
  ])
);
let ASSET_ROOT = ASSET_ROOT_CANDIDATES[0];
let MODEL_PATH = "";
let META_PATH = "";
let PRESETS_PATH = "";
let VOCAB_PATH = "";
let MERGES_PATH = "";
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
  VOCAB_PATH = `${ASSET_ROOT}/vocab.json`;
  MERGES_PATH = `${ASSET_ROOT}/merges.txt`;
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
let modelMeta = null;
let promptPresets = {};
let tokenizer = null;
let running = false;
let inputTypeByName = {};
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
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
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
  return Boolean(meta?.weights?.trained);
}

function setMissingWeightsState(message) {
  const detail = message ? ` (${message})` : "";
  weightsBadge.textContent = `Weights: missing${detail}`;
  providerBadge.textContent = "Provider: unavailable";
  modelStatus.textContent = "Model: unavailable (missing weights)";
  setGenerationEnabled(false);
}

function contextSize() {
  return toNumber(modelMeta?.generation?.context_size, 1024);
}

function eosTokenId() {
  return toNumber(modelMeta?.tokenizer?.eos_token_id, 50256);
}

function loadJson(path) {
  return fetch(path, { cache: "no-store" }).then((response) => {
    if (!response.ok) {
      throw new Error(`failed to fetch ${path}: HTTP ${response.status}`);
    }
    return response.json();
  });
}

function loadText(path) {
  return fetch(path, { cache: "no-store" }).then((response) => {
    if (!response.ok) {
      throw new Error(`failed to fetch ${path}: HTTP ${response.status}`);
    }
    return response.text();
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

function bytesToUnicodeMap() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const out = new Map();
  for (let i = 0; i < bs.length; i += 1) {
    out.set(bs[i], String.fromCodePoint(cs[i]));
  }
  return out;
}

class Gpt2BpeTokenizer {
  constructor(vocab, mergesText) {
    this.encoder = vocab;
    this.decoder = {};
    Object.entries(vocab).forEach(([token, id]) => {
      this.decoder[Number(id)] = token;
    });

    this.byteEncoder = bytesToUnicodeMap();
    this.byteDecoder = new Map();
    this.byteEncoder.forEach((value, key) => {
      this.byteDecoder.set(value, key);
    });

    const merges = mergesText
      .split(/\r?\n/u)
      .slice(1)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => line.split(" "));

    this.bpeRanks = new Map();
    merges.forEach((pair, idx) => {
      this.bpeRanks.set(`${pair[0]} ${pair[1]}`, idx);
    });

    this.cache = new Map();
    this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.textEncoder = new TextEncoder();
    this.textDecoder = new TextDecoder("utf-8", { fatal: false });
  }

  getPairs(word) {
    const pairs = new Set();
    if (word.length < 2) {
      return pairs;
    }
    let prev = word[0];
    for (let i = 1; i < word.length; i += 1) {
      const current = word[i];
      pairs.add(`${prev}\u0001${current}`);
      prev = current;
    }
    return pairs;
  }

  bpe(token) {
    if (this.cache.has(token)) {
      return this.cache.get(token);
    }

    let word = Array.from(token);
    if (word.length === 1) {
      this.cache.set(token, token);
      return token;
    }

    let pairs = this.getPairs(word);

    while (pairs.size > 0) {
      let minPair = null;
      let minRank = Infinity;

      pairs.forEach((pairKey) => {
        const pair = pairKey.split("\u0001");
        const rank = this.bpeRanks.get(`${pair[0]} ${pair[1]}`);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      });

      if (minPair === null) {
        break;
      }

      const [first, second] = minPair;
      const nextWord = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          nextWord.push(...word.slice(i));
          break;
        }
        nextWord.push(...word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          nextWord.push(first + second);
          i += 2;
        } else {
          nextWord.push(word[i]);
          i += 1;
        }
      }

      word = nextWord;
      if (word.length === 1) {
        break;
      }
      pairs = this.getPairs(word);
    }

    const output = word.join(" ");
    this.cache.set(token, output);
    return output;
  }

  encode(text) {
    const ids = [];
    const matches = String(text || "").match(this.pattern) || [];

    matches.forEach((fragment) => {
      const encoded = Array.from(this.textEncoder.encode(fragment))
        .map((b) => this.byteEncoder.get(b))
        .join("");

      this.bpe(encoded)
        .split(" ")
        .forEach((token) => {
          const id = this.encoder[token];
          if (id !== undefined) {
            ids.push(Number(id));
          }
        });
    });

    return ids;
  }

  decode(ids) {
    const merged = ids
      .map((id) => this.decoder[Number(id)] || "")
      .join("");

    const bytes = [];
    for (const ch of Array.from(merged)) {
      if (this.byteDecoder.has(ch)) {
        bytes.push(this.byteDecoder.get(ch));
      }
    }

    return this.textDecoder.decode(Uint8Array.from(bytes));
  }

  tokenLabel(id) {
    const token = this.decoder[Number(id)] || "<unk>";
    return JSON.stringify(token);
  }
}

function makeTensor(values, type, dims) {
  const typeLabel = String(type || "").toLowerCase();
  if (typeLabel.includes("bool")) {
    const normalized = values.map((value) => (value ? 1 : 0));
    return new ort.Tensor("bool", Uint8Array.from(normalized), dims);
  }
  if (typeLabel.includes("float")) {
    return new ort.Tensor("float32", Float32Array.from(values), dims);
  }
  if (typeLabel.includes("64")) {
    return new ort.Tensor("int64", BigInt64Array.from(values.map((v) => BigInt(v))), dims);
  }
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}

function inputType(name, fallback = "int64") {
  if (name in inputTypeByName) {
    return inputTypeByName[name];
  }
  return fallback;
}

function createFeeds(tokens) {
  const feeds = {};
  const inputNames = session.inputNames || [];

  inputNames.forEach((name) => {
    const type = inputType(name, "int64");

    if (name.includes("input_ids")) {
      feeds[name] = makeTensor(tokens, type, [1, tokens.length]);
      return;
    }

    if (name.includes("attention_mask")) {
      feeds[name] = makeTensor(Array(tokens.length).fill(1), type, [1, tokens.length]);
      return;
    }

    if (name.includes("position_ids")) {
      const positions = Array.from({ length: tokens.length }, (_, i) => i);
      feeds[name] = makeTensor(positions, type, [1, tokens.length]);
      return;
    }

    if (name.includes("past_key_values")) {
      feeds[name] = makeTensor([], type, [1, 12, 0, 64]);
      return;
    }

    if (name.includes("use_cache_branch")) {
      feeds[name] = makeTensor([false], type, [1]);
    }
  });

  // Exported Ruby GPT-2 ONNX usually has a single token input.
  if (inputNames.length === 1 && !(inputNames[0] in feeds)) {
    const name = inputNames[0];
    const type = inputType(name, "int32");
    feeds[name] = makeTensor(tokens, type, [1, tokens.length]);
  }

  return feeds;
}

function outputTensor(outputs) {
  const logitsName = (session.outputNames || []).find((name) => name.includes("logits"));
  if (logitsName && logitsName in outputs) {
    return outputs[logitsName];
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
  const sequenceLength = dims[1];
  const vocabSize = dims[2];
  const start = (sequenceLength - 1) * vocabSize;
  const end = start + vocabSize;
  return tensor.data.slice(start, end);
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number(values[0]);
  for (let i = 1; i < values.length; i += 1) {
    const value = Number(values[i]);
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
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
      <td>${tokenizer ? tokenizer.tokenLabel(entry.id) : "<unk>"}</td>
      <td>${(entry.prob * 100).toFixed(2)}%</td>
      <td>${entry.logit.toFixed(4)}</td>
    `;
    topKBody.appendChild(row);
  });
}

function setTemperatureLabel() {
  temperatureValue.textContent = toNumber(temperatureInput.value, 0.8).toFixed(2);
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

function encodedPrompt(prompt) {
  const ids = tokenizer ? tokenizer.encode(prompt) : [];
  return ids.length > 0 ? ids : [eosTokenId()];
}

function contextWindow(tokenIds) {
  const window = Math.max(1, contextSize());
  const trimmed = tokenIds.slice(-window);
  if (trimmed.length >= window) {
    return trimmed;
  }
  return Array(window - trimmed.length).fill(eosTokenId()).concat(trimmed);
}

async function runGeneration() {
  if (!session || !modelMeta || !tokenizer || running) {
    return;
  }

  running = true;
  setGenerationEnabled(false);
  clearError();

  try {
    const temperature = Math.max(0.0, toNumber(temperatureInput.value, 0.8));
    const maxTokens = Math.max(1, Math.min(256, toNumber(maxTokensInput.value, 80)));
    const prompt = String(promptInput.value || "");
    const promptIds = encodedPrompt(prompt);
    const generatedIds = [];
    const allIds = [...promptIds];

    contextStatus.textContent = `Prompt tokens: ${promptIds.length}, context window: ${contextSize()}`;

    let firstStepLogits = null;
    let totalMs = 0.0;

    for (let step = 0; step < maxTokens; step += 1) {
      const context = contextWindow(allIds);
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
      allIds.push(nextToken);
      if (nextToken === eosTokenId()) {
        break;
      }
    }

    renderTopK(firstStepLogits || []);
    outputText.textContent = tokenizer.decode(allIds);
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
      contextStatus.textContent = "Prompt tokens: unavailable";
      shapeStatus.textContent = "Output shape: unavailable";
      showError(missingAssetsMessage());
      return;
    }
    setOnnxSizeText("ONNX Size: probing...");
    const onnxSizeProbe = updateOnnxSizeStatus();
    modelStatus.textContent = "Runtime: loading ONNX Runtime Web...";
    ort = await loadOrtModule();
    modelStatus.textContent = "Model: loading metadata...";
    const [meta, presets, vocab, merges] = await Promise.all([
      loadJson(META_PATH),
      loadJson(PRESETS_PATH),
      loadJson(VOCAB_PATH),
      loadText(MERGES_PATH)
    ]);
    await onnxSizeProbe;

    modelMeta = meta;
    promptPresets = presets;
    tokenizer = new Gpt2BpeTokenizer(vocab, merges);
    setParameterText(modelMeta?.parameters?.total ?? modelMeta?.parameter_count);
    inputTypeByName = {};
    (modelMeta.inputs || []).forEach((entry) => {
      if (entry && entry.name) {
        inputTypeByName[entry.name] = String(entry.type || "");
      }
    });

    installPresets();
    installUi();
    setTemperatureLabel();
    setMaxTokensLabel();

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
      contextStatus.textContent = "Prompt tokens: unavailable";
      shapeStatus.textContent = "Output shape: unavailable";
      return;
    }

    modelStatus.textContent = "Model: creating runtime session...";
    const created = await createSessionWithFallback();
    session = created.session;

    providerBadge.textContent = `Provider: ${created.provider}`;
    if (created.provider !== "webgpu") {
      providerBadge.textContent += " (fallback)";
    }

    const repoId = String(modelMeta?.weights?.repo_id || "unknown");
    weightsBadge.textContent = `Weights: ${repoId} via MLX Ruby export`;
    tokenizerStatus.textContent = `Tokenizer: GPT-2 BPE (vocab=${toNumber(modelMeta?.tokenizer?.vocab_size, 0)})`;
    modelStatus.textContent = `Model: ${modelMeta.model_name}`;
    contextStatus.textContent = "Prompt tokens: ready";
    setGenerationEnabled(true);

    const outputName = (session.outputNames || [])[0] || "<unknown>";
    const outputMeta = session.outputMetadata?.[outputName];
    shapeStatus.textContent = `Output: ${outputName} ${JSON.stringify(outputMeta?.dimensions || [])}`;
  } catch (error) {
    showError(String(error));
    setMissingWeightsState("unavailable");
    providerBadge.textContent = "Provider: unavailable";
    setParameterText(null);
    modelStatus.textContent = "Model: failed to load";
    setGenerationEnabled(false);
  }
}

boot();
