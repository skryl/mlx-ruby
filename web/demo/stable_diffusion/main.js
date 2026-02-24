const ASSET_ROOT_CANDIDATES = Array.from(
  new Set([
    new URL("../../assets/stable_diffusion", import.meta.url).toString().replace(/\/$/, ""),
    new URL("../assets/stable_diffusion", import.meta.url).toString().replace(/\/$/, ""),
    new URL("./assets/stable_diffusion", import.meta.url).toString().replace(/\/$/, "")
  ])
);
let ASSET_ROOT = ASSET_ROOT_CANDIDATES[0];
let META_PATH = "";
let PRESETS_PATH = "";
let SCHEDULER_CONFIG_PATH = "";
let VOCAB_PATH = "";
let MERGES_PATH = "";
let TEXT_ENCODER_PATH = "";
let UNET_PATH = "";
let VAE_DECODER_PATH = "";
const ORT_MODULE_CANDIDATES = [
  "../../node_modules/onnxruntime-web/dist/ort.all.min.mjs",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs"
];

setAssetRoot(ASSET_ROOT);

function setAssetRoot(root) {
  ASSET_ROOT = root;
  META_PATH = `${ASSET_ROOT}/meta.json`;
  PRESETS_PATH = `${ASSET_ROOT}/prompt.presets.json`;
  SCHEDULER_CONFIG_PATH = `${ASSET_ROOT}/scheduler_config.json`;
  VOCAB_PATH = `${ASSET_ROOT}/vocab.json`;
  MERGES_PATH = `${ASSET_ROOT}/merges.txt`;
  TEXT_ENCODER_PATH = `${ASSET_ROOT}/text_encoder.onnx`;
  UNET_PATH = `${ASSET_ROOT}/unet.onnx`;
  VAE_DECODER_PATH = `${ASSET_ROOT}/vae_decoder.onnx`;
}

const weightsBadge = document.getElementById("badge-weights");
const providerBadge = document.getElementById("badge-provider");
const onnxSizeBadge = document.getElementById("badge-onnx-size");
const parametersBadge = document.getElementById("badge-parameters");
const timingBadge = document.getElementById("badge-timing");

const modelStatus = document.getElementById("status-model");
const ioStatus = document.getElementById("status-io");
const outputStatus = document.getElementById("status-output");
const statsStatus = document.getElementById("status-stats");
const onnxSizeStatus = document.getElementById("status-onnx-size");
const previewValues = document.getElementById("preview-values");
const errorBox = document.getElementById("error-box");

const presetSelect = document.getElementById("preset");
const presetName = document.getElementById("preset-name");
const promptInput = document.getElementById("prompt");
const seedInput = document.getElementById("seed");
const timestepInput = document.getElementById("timestep");
const timestepValue = document.getElementById("timestep-value");
const stepsInput = document.getElementById("steps");
const stepsValue = document.getElementById("steps-value");
const guidanceInput = document.getElementById("guidance");
const guidanceValue = document.getElementById("guidance-value");
const runButton = document.getElementById("run");
const newSeedButton = document.getElementById("new-seed");
const canvas = document.getElementById("latent-canvas");
const generatedCanvas = document.getElementById("generated-canvas");

const canvasContext = canvas.getContext("2d");
const generatedCanvasContext = generatedCanvas.getContext("2d");

let modelMeta = null;
let promptPresets = {};
let schedulerConfig = {};
let pipelineSessions = null;
let tokenizer = null;
let selectedProviders = null;
let running = false;
let ort = null;

runButton.disabled = true;

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
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setRunEnabled(enabled) {
  runButton.disabled = !enabled;
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
  const specs = [
    { label: "text", path: TEXT_ENCODER_PATH },
    { label: "unet", path: UNET_PATH },
    { label: "vae", path: VAE_DECODER_PATH }
  ];

  const measured = await Promise.all(specs.map(async (spec) => {
    const bytes = await onnxSizeBytes(spec.path);
    return { ...spec, bytes };
  }));

  const available = measured.filter((entry) => Number.isFinite(entry.bytes));
  if (available.length === 0) {
    setOnnxSizeText("ONNX Size: unavailable");
    return;
  }

  const totalBytes = available.reduce((sum, entry) => sum + entry.bytes, 0);
  const parts = measured.map((entry) => {
    if (!Number.isFinite(entry.bytes)) {
      return `${entry.label}=missing`;
    }
    return `${entry.label}=${formatByteSize(entry.bytes)}`;
  });
  setOnnxSizeText(`ONNX Size: ${parts.join(", ")} (total ${formatByteSize(totalBytes)})`);
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

function basename(path) {
  const normalized = String(path || "");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || "";
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

async function sessionOptionsForModel(provider, modelPath) {
  const options = { executionProviders: [provider] };
  const modelFile = basename(modelPath);
  let stageSpec = null;

  if (modelFile === "text_encoder.onnx") {
    stageSpec = pipelineSpec()?.text_encoder || null;
  } else if (modelFile === "unet.onnx") {
    stageSpec = pipelineSpec()?.unet || null;
  } else if (modelFile === "vae_decoder.onnx") {
    stageSpec = pipelineSpec()?.vae_decoder || null;
  }

  if (stageSpec && Object.prototype.hasOwnProperty.call(stageSpec, "external_data")) {
    const externalDataName = stageSpec.external_data;
    if (typeof externalDataName === "string" && externalDataName.length > 0) {
      options.externalData = [
        {
          path: externalDataName,
          data: `${ASSET_ROOT}/${externalDataName}`
        }
      ];
    }
    return options;
  }

  const externalDataPath = `${modelPath}.data`;
  if (await assetExists(externalDataPath)) {
    options.externalData = [
      {
        path: `${basename(modelPath)}.data`,
        data: externalDataPath
      }
    ];
  }
  return options;
}

async function createPipelineSessions() {
  const textUnetProviders = ["webgpu", "wasm"];
  const vaeProviders = ["webgpu", "wasm"];
  const errors = [];

  for (const textUnetProvider of textUnetProviders) {
    try {
      const [textEncoderOptions, unetOptions] = await Promise.all([
        sessionOptionsForModel(textUnetProvider, TEXT_ENCODER_PATH),
        sessionOptionsForModel(textUnetProvider, UNET_PATH)
      ]);

      const [textEncoder, unet] = await Promise.all([
        ort.InferenceSession.create(TEXT_ENCODER_PATH, textEncoderOptions),
        ort.InferenceSession.create(UNET_PATH, unetOptions)
      ]);

      for (const vaeProvider of vaeProviders) {
        try {
          const vaeDecoderOptions = await sessionOptionsForModel(vaeProvider, VAE_DECODER_PATH);
          const vaeDecoder = await ort.InferenceSession.create(VAE_DECODER_PATH, vaeDecoderOptions);
          return {
            sessions: { textEncoder, unet, vaeDecoder },
            providers: {
              textUnet: textUnetProvider,
              vaeDecoder: vaeProvider
            },
            errors
          };
        } catch (error) {
          errors.push({ provider: `vae:${vaeProvider}`, message: String(error) });
        }
      }
    } catch (error) {
      errors.push({ provider: `text+unet:${textUnetProvider}`, message: String(error) });
    }
  }

  throw new Error(
    `failed to initialize stable diffusion pipeline sessions:\n${errors
      .map((entry) => `- ${entry.provider}: ${entry.message}`)
      .join("\n")}`
  );
}

function shapeProduct(shape) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return 0;
  }
  return shape.reduce((acc, value) => acc * Math.max(1, toNumber(value, 1)), 1);
}

function normalizeShape(shape, fallback) {
  if (!Array.isArray(shape) || shape.length === 0) {
    return fallback;
  }
  return shape.map((entry, idx) => {
    const parsed = toNumber(entry, NaN);
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.trunc(parsed);
    }
    return fallback[idx] ?? 1;
  });
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

class ClipBpeTokenizer {
  constructor(vocab, mergesText, config = {}) {
    this.encoder = vocab;
    this.byteEncoder = bytesToUnicodeMap();
    this.byteDecoder = new Map();
    this.byteEncoder.forEach((value, key) => {
      this.byteDecoder.set(value, key);
    });

    const merges = mergesText
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"))
      .map((line) => line.split(/\s+/u));

    this.bpeRanks = new Map();
    merges.forEach((pair, idx) => {
      if (pair.length === 2) {
        this.bpeRanks.set(`${pair[0]} ${pair[1]}`, idx);
      }
    });

    this.cache = new Map();
    this.pattern = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.textEncoder = new TextEncoder();

    this.maxLength = toNumber(config.max_length, 77);
    this.bosTokenId = toNumber(config.bos_token_id, 49406);
    this.eosTokenId = toNumber(config.eos_token_id, 49407);
    this.padTokenId = toNumber(config.pad_token_id, this.eosTokenId);
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
    if (word.length === 0) {
      return "";
    }
    word[word.length - 1] = `${word[word.length - 1]}</w>`;

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
    const tokens = [];
    const normalized = String(text || "").toLowerCase().trim();
    const matches = normalized.match(this.pattern) || [];

    matches.forEach((fragment) => {
      const encoded = Array.from(this.textEncoder.encode(fragment))
        .map((b) => this.byteEncoder.get(b))
        .join("");

      this.bpe(encoded)
        .split(" ")
        .forEach((token) => {
          const id = this.encoder[token];
          if (id !== undefined) {
            tokens.push(Number(id));
          }
        });
    });

    return tokens;
  }

  encodePrompt(prompt) {
    const body = this.encode(prompt);
    const full = [this.bosTokenId, ...body, this.eosTokenId];
    const sliced = full.slice(0, this.maxLength);
    if (sliced.length < this.maxLength) {
      return sliced.concat(Array(this.maxLength - sliced.length).fill(this.padTokenId));
    }
    return sliced;
  }
}

async function loadTokenizer() {
  const [vocab, merges] = await Promise.all([loadJson(VOCAB_PATH), loadText(MERGES_PATH)]);
  tokenizer = new ClipBpeTokenizer(vocab, merges, modelMeta?.tokenizer || {});
}

function pipelineSpec() {
  return modelMeta?.pipeline || {};
}

function stepsRange() {
  const min = Math.max(1, Math.trunc(toNumber(modelMeta?.generation?.recommended_steps_min, 1)));
  const max = Math.max(min, Math.trunc(toNumber(modelMeta?.generation?.recommended_steps_max, 64)));
  return { min, max };
}

function guidanceRange() {
  const min = Math.max(0.0, toNumber(modelMeta?.generation?.recommended_guidance_min, 0.0));
  const max = Math.max(min, toNumber(modelMeta?.generation?.recommended_guidance_max, 14.0));
  return { min, max };
}

function inputSpec(stage, index) {
  return pipelineSpec()?.[stage]?.inputs?.[index] || null;
}

function unetInputShape() {
  const spec = inputSpec("unet", 0);
  return normalizeShape(spec?.shape, [1, 4, 32, 32]);
}

function vaeInputShape() {
  const spec = inputSpec("vae_decoder", 0);
  return normalizeShape(spec?.shape, unetInputShape());
}

function tensorTypeForSpec(spec, fallback = "float32") {
  const raw = String(spec?.type || fallback).toLowerCase();
  if (raw.includes("int64") || raw.includes("tensor(int64)")) return "int64";
  if (raw.includes("int32") || raw.includes("tensor(int32)")) return "int32";
  if (raw.includes("bool") || raw.includes("tensor(bool)")) return "bool";
  if (raw.includes("float16") || raw.includes("tensor(float16)")) return "float16";
  return "float32";
}

const FLOAT32_BUFFER = new Float32Array(1);
const UINT32_BUFFER = new Uint32Array(FLOAT32_BUFFER.buffer);

function float16ToFloat32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign * 0.0;
    }
    return sign * (fraction / 1024) * (2 ** -14);
  }

  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }

  return sign * (1 + (fraction / 1024)) * (2 ** (exponent - 15));
}

function float32ToFloat16Bits(value) {
  FLOAT32_BUFFER[0] = Number(value);
  const bits = UINT32_BUFFER[0];

  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127;
  let mantissa = bits & 0x007fffff;

  if (exponent === 128) {
    if (mantissa !== 0) {
      return sign | 0x7e00;
    }
    return sign | 0x7c00;
  }

  if (exponent > 15) {
    return sign | 0x7c00;
  }

  if (exponent < -24) {
    return sign;
  }

  if (exponent < -14) {
    mantissa |= 0x00800000;
    const shift = -exponent - 14;
    let halfMantissa = mantissa >> (shift + 13);
    if (((mantissa >> (shift + 12)) & 1) === 1) {
      halfMantissa += 1;
    }
    return sign | (halfMantissa & 0x03ff);
  }

  exponent += 15;
  let halfMantissa = mantissa >> 13;
  if ((mantissa & 0x00001000) !== 0) {
    halfMantissa += 1;
    if ((halfMantissa & 0x0400) !== 0) {
      halfMantissa = 0;
      exponent += 1;
      if (exponent >= 31) {
        return sign | 0x7c00;
      }
    }
  }

  return sign | (exponent << 10) | (halfMantissa & 0x03ff);
}

function tensorValueAt(data, index, tensorType = "float32") {
  if (tensorType === "float16") {
    return float16ToFloat32(Number(data[index]));
  }
  return Number(data[index]);
}

function makeTensor(values, spec, dimsFallback) {
  const dims = normalizeShape(spec?.shape, dimsFallback);
  const type = tensorTypeForSpec(spec, "float32");

  if (type === "int64") {
    const casted = BigInt64Array.from(Array.from(values, (value) => BigInt(Math.trunc(Number(value)))));
    return new ort.Tensor("int64", casted, dims);
  }

  if (type === "int32") {
    const casted = Int32Array.from(Array.from(values, (value) => Math.trunc(Number(value))));
    return new ort.Tensor("int32", casted, dims);
  }

  if (type === "bool") {
    const casted = Uint8Array.from(Array.from(values, (value) => (value ? 1 : 0)));
    return new ort.Tensor("bool", casted, dims);
  }

  if (type === "float16") {
    const casted = Uint16Array.from(Array.from(values, (value) => float32ToFloat16Bits(value)));
    return new ort.Tensor("float16", casted, dims);
  }

  const casted = Float32Array.from(Array.from(values, (value) => Number(value)));
  return new ort.Tensor("float32", casted, dims);
}

function summarizeTensor(tensorLike) {
  const data = tensorLike?.data || tensorLike;
  const tensorType = String(tensorLike?.type || "float32").toLowerCase();
  if (!data || data.length === 0) {
    return "Stats: empty tensor";
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0.0;
  for (let i = 0; i < data.length; i += 1) {
    const value = tensorValueAt(data, i, tensorType);
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }

  const mean = sum / data.length;
  return `Stats: min=${min.toFixed(5)} max=${max.toFixed(5)} mean=${mean.toFixed(5)}`;
}

function previewTensorValues(tensorLike, count = 18) {
  const data = tensorLike?.data || tensorLike;
  const tensorType = String(tensorLike?.type || "float32").toLowerCase();
  if (!data) return "--";

  const values = [];
  for (let i = 0; i < Math.min(count, data.length); i += 1) {
    values.push(tensorValueAt(data, i, tensorType).toFixed(6));
  }
  return values.join(", ");
}

function tensorReader(tensor) {
  const dims = tensor?.dims || [];
  const data = tensor?.data;
  const tensorType = String(tensor?.type || "float32").toLowerCase();
  if (!data || dims.length !== 4) {
    return null;
  }

  if (dims[3] > 0 && dims[3] <= 8) {
    const height = dims[1];
    const width = dims[2];
    const channels = dims[3];
    return {
      width,
      height,
      channels,
      at: (x, y, channel) => tensorValueAt(
        data,
        ((y * width) + x) * channels + Math.min(channel, channels - 1),
        tensorType
      )
    };
  }

  if (dims[1] > 0 && dims[1] <= 8) {
    const channels = dims[1];
    const height = dims[2];
    const width = dims[3];
    return {
      width,
      height,
      channels,
      at: (x, y, channel) => tensorValueAt(
        data,
        (Math.min(channel, channels - 1) * height * width) + (y * width) + x,
        tensorType
      )
    };
  }

  return null;
}

function renderTensorToCanvas(tensor, targetCanvas, targetContext, { detailMix = 0.0, gamma = 1.0, preferUnitRange = false } = {}) {
  const reader = tensorReader(tensor);
  if (!reader) {
    return false;
  }

  const { width, height } = reader;
  targetCanvas.width = width;
  targetCanvas.height = height;

  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        const base = Number(reader.at(x, y, c));
        const detail = Number(reader.at(x, y, 3)) * detailMix;
        const value = base + detail;
        if (value < mins[c]) mins[c] = value;
        if (value > maxs[c]) maxs[c] = value;
      }
    }
  }

  const imageData = targetContext.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      for (let c = 0; c < 3; c += 1) {
        const base = Number(reader.at(x, y, c));
        const detail = Number(reader.at(x, y, 3)) * detailMix;
        const raw = base + detail;

        let normalized;
        if (preferUnitRange && mins[c] >= -1.001 && maxs[c] <= 1.001) {
          if (mins[c] >= -0.001 && maxs[c] <= 1.001) {
            normalized = raw;
          } else {
            normalized = (raw + 1.0) * 0.5;
          }
        } else {
          const range = Math.max(1e-8, maxs[c] - mins[c]);
          normalized = (raw - mins[c]) / range;
        }

        const color = Math.pow(clamp(normalized, 0.0, 1.0), gamma);
        imageData.data[idx + c] = Math.round(color * 255);
      }
      imageData.data[idx + 3] = 255;
    }
  }

  targetContext.putImageData(imageData, 0, 0);
  return true;
}

function renderLatentPreview(latents, shape) {
  const tensor = {
    data: latents,
    dims: shape,
    type: "float32"
  };

  const rendered = renderTensorToCanvas(tensor, canvas, canvasContext, { detailMix: 0.0, gamma: 1.0, preferUnitRange: false });
  if (!rendered) {
    outputStatus.textContent = `Latent: unsupported layout ${JSON.stringify(shape)}`;
  }
}

function renderGeneratedImage(tensor) {
  const rendered = renderTensorToCanvas(tensor, generatedCanvas, generatedCanvasContext, {
    detailMix: 0.0,
    gamma: 1.0,
    preferUnitRange: true
  });
  if (!rendered) {
    outputStatus.textContent = `Output: unsupported tensor layout ${JSON.stringify(tensor?.dims || [])}`;
  }
}

function schedulerSettings() {
  const numTrainTimesteps = Math.max(
    10,
    Math.trunc(
      toNumber(
        schedulerConfig?.num_train_timesteps,
        toNumber(modelMeta?.generation?.max_time, 999) + 1
      )
    )
  );
  const betaStart = toNumber(schedulerConfig?.beta_start, 0.00085);
  const betaEnd = toNumber(schedulerConfig?.beta_end, 0.012);
  return { numTrainTimesteps, betaStart, betaEnd };
}

function buildAlphaCumprod() {
  const { numTrainTimesteps, betaStart, betaEnd } = schedulerSettings();
  const alphasCumprod = new Float64Array(numTrainTimesteps);
  let running = 1.0;
  for (let i = 0; i < numTrainTimesteps; i += 1) {
    const ratio = numTrainTimesteps > 1 ? (i / (numTrainTimesteps - 1)) : 0;
    const beta = betaStart + ((betaEnd - betaStart) * ratio);
    const alpha = 1.0 - beta;
    running *= alpha;
    alphasCumprod[i] = running;
  }
  return alphasCumprod;
}

function maxTime() {
  const fromMeta = toNumber(modelMeta?.generation?.max_time, NaN);
  if (Number.isFinite(fromMeta) && fromMeta > 0) {
    return Math.trunc(fromMeta);
  }
  const { numTrainTimesteps } = schedulerSettings();
  return Math.max(1, numTrainTimesteps - 1);
}

function timestepsForRun(steps, startTimestep) {
  const count = Math.max(1, Math.trunc(steps));
  const start = clamp(Math.trunc(startTimestep), 1, maxTime());
  const values = [];
  for (let i = 0; i <= count; i += 1) {
    const ratio = i / count;
    values.push(Math.max(0, Math.round(start * (1.0 - ratio))));
  }
  values[count] = 0;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > values[i - 1]) {
      values[i] = values[i - 1];
    }
  }
  return values;
}

function blendGuidedNoise(condNoise, uncondNoise, guidance) {
  const out = new Float32Array(condNoise.length);
  for (let i = 0; i < condNoise.length; i += 1) {
    out[i] = uncondNoise[i] + (guidance * (condNoise[i] - uncondNoise[i]));
  }
  return out;
}

function schedulerStep(latents, epsPred, tIndex, tPrevIndex, alphasCumprod) {
  const alphaT = alphasCumprod[Math.max(0, Math.min(alphasCumprod.length - 1, tIndex))];
  const alphaPrev = tPrevIndex > 0
    ? alphasCumprod[Math.max(0, Math.min(alphasCumprod.length - 1, tPrevIndex))]
    : 1.0;

  const sqrtAlphaT = Math.sqrt(Math.max(1e-8, alphaT));
  const sqrtOneMinusAlphaT = Math.sqrt(Math.max(1e-8, 1.0 - alphaT));
  const sqrtAlphaPrev = Math.sqrt(Math.max(1e-8, alphaPrev));
  const sqrtOneMinusAlphaPrev = Math.sqrt(Math.max(0.0, 1.0 - alphaPrev));

  const next = new Float32Array(latents.length);
  for (let i = 0; i < latents.length; i += 1) {
    const predX0 = (latents[i] - (sqrtOneMinusAlphaT * epsPred[i])) / sqrtAlphaT;
    next[i] = clamp((sqrtAlphaPrev * predX0) + (sqrtOneMinusAlphaPrev * epsPred[i]), -20.0, 20.0);
  }
  return next;
}

function resolveFirstOutput(outputs) {
  const values = Object.values(outputs);
  if (values.length === 0) {
    throw new Error("runtime returned no outputs");
  }
  return values[0];
}

function hashString(value) {
  let hash = 2166136261;
  const text = String(value || "");
  for (let idx = 0; idx < text.length; idx += 1) {
    hash ^= text.charCodeAt(idx);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let t = Math.imul(value ^ (value >>> 15), value | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildLatents(seed, shape) {
  const size = shapeProduct(shape);
  const rng = mulberry32(seed);
  const output = new Float32Array(size);
  for (let i = 0; i < size; i += 1) {
    output[i] = (rng() * 2.0) - 1.0;
  }
  return output;
}

async function runTextEncoder(prompt) {
  const session = pipelineSessions.textEncoder;
  const tokenIds = tokenizer.encodePrompt(prompt);
  const spec = inputSpec("text_encoder", 0);
  const feeds = {
    [session.inputNames[0]]: makeTensor(tokenIds, spec, [1, tokenIds.length])
  };
  const outputs = await session.run(feeds);
  return resolveFirstOutput(outputs);
}

async function runUnet(latents, timestep, conditioning) {
  const session = pipelineSessions.unet;
  const sampleSpec = inputSpec("unet", 0);
  const timestepSpec = inputSpec("unet", 1);
  const conditioningSpec = inputSpec("unet", 2);

  const feeds = {
    [session.inputNames[0]]: makeTensor(latents, sampleSpec, unetInputShape()),
    [session.inputNames[1]]: makeTensor([Math.trunc(timestep)], timestepSpec, [1]),
    [session.inputNames[2]]: makeTensor(conditioning.data, conditioningSpec, conditioning.dims)
  };

  const outputs = await session.run(feeds);
  return resolveFirstOutput(outputs);
}

async function runDecoder(latents) {
  const session = pipelineSessions.vaeDecoder;
  const spec = inputSpec("vae_decoder", 0);
  const scalingFactor = Math.max(1e-8, toNumber(modelMeta?.generation?.vae_scaling_factor, 0.18215));
  const scaledLatents = Float32Array.from(latents, (value) => Number(value) / scalingFactor);

  const feeds = {
    [session.inputNames[0]]: makeTensor(scaledLatents, spec, vaeInputShape())
  };

  const outputs = await session.run(feeds);
  return resolveFirstOutput(outputs);
}

async function runMultiStepDenoise() {
  if (!pipelineSessions || !tokenizer || running) {
    return;
  }

  running = true;
  setRunEnabled(false);
  clearError();

  try {
    const prompt = promptInput.value || "";
    const seed = Math.max(0, Math.trunc(toNumber(seedInput.value, 1337)));
    seedInput.value = String(seed);
    const stepLimits = stepsRange();
    const guidanceLimits = guidanceRange();

    const steps = Math.max(stepLimits.min, Math.min(stepLimits.max, Math.trunc(toNumber(stepsInput.value, 20))));
    const guidance = clamp(toNumber(guidanceInput.value, 7.5), guidanceLimits.min, guidanceLimits.max);
    const startTimestep = clamp(toNumber(timestepInput.value, maxTime()), 1.0, maxTime());

    stepsInput.value = String(steps);
    guidanceInput.value = guidance.toFixed(2);
    timestepInput.value = startTimestep.toFixed(0);
    setStepsLabel();
    setGuidanceLabel();
    setTimestepLabel();

    const latentShape = unetInputShape();
    let latents = buildLatents(seed ^ hashString(prompt), latentShape);
    renderLatentPreview(latents, latentShape);

    const cond = await runTextEncoder(prompt);
    const uncond = guidance > 1.0001 ? await runTextEncoder("") : null;

    const tValues = timestepsForRun(steps, startTimestep);
    const alphasCumprod = buildAlphaCumprod();
    const startedAt = performance.now();

    for (let stepIndex = 0; stepIndex < steps; stepIndex += 1) {
      const t = tValues[stepIndex];
      const tPrev = tValues[stepIndex + 1];
      outputStatus.textContent = `Denoising ${stepIndex + 1}/${steps} (t=${t})`;

      const condTensor = await runUnet(latents, t, cond);
      const condNoise = Float32Array.from(condTensor.data, (value) => Number(value));

      let epsPred = condNoise;
      if (uncond) {
        const uncondTensor = await runUnet(latents, t, uncond);
        const uncondNoise = Float32Array.from(uncondTensor.data, (value) => Number(value));
        epsPred = blendGuidedNoise(condNoise, uncondNoise, guidance);
      }

      latents = schedulerStep(latents, epsPred, t, tPrev, alphasCumprod);
      renderLatentPreview(latents, latentShape);
    }

    const decodedTensor = await runDecoder(latents);
    const elapsed = performance.now() - startedAt;

    renderGeneratedImage(decodedTensor);
    timingBadge.textContent = `Inference: ${elapsed.toFixed(2)} ms`;
    outputStatus.textContent = `Output: ${JSON.stringify(decodedTensor.dims)} (${decodedTensor.type})`;
    statsStatus.textContent = `${summarizeTensor(decodedTensor)} steps=${steps} guidance=${guidance.toFixed(2)}`;
    previewValues.textContent = previewTensorValues(decodedTensor);
  } catch (error) {
    showError(String(error));
  } finally {
    running = false;
    setRunEnabled(true);
  }
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

function setTimestepLabel() {
  timestepValue.textContent = toNumber(timestepInput.value, 1).toFixed(0);
}

function setStepsLabel() {
  const limits = stepsRange();
  const steps = Math.max(limits.min, Math.min(limits.max, Math.trunc(toNumber(stepsInput.value, 20))));
  stepsInput.value = String(steps);
  stepsValue.textContent = String(steps);
}

function setGuidanceLabel() {
  const limits = guidanceRange();
  const guidance = clamp(toNumber(guidanceInput.value, 7.5), limits.min, limits.max);
  guidanceInput.value = guidance.toFixed(2);
  guidanceValue.textContent = guidance.toFixed(2);
}

function randomizeSeed() {
  const next = Math.floor(Math.random() * 0x7fffffff);
  seedInput.value = String(next);
}

function updateStatuses() {
  modelStatus.textContent = `Model: ${modelMeta?.model_name || "unknown"}`;
  const unet = pipelineSpec()?.unet || {};
  const decoder = pipelineSpec()?.vae_decoder || {};
  ioStatus.textContent = `I/O: unet ${JSON.stringify(unet.inputs?.[0]?.shape || [])} -> ${JSON.stringify(unet.outputs?.[0]?.shape || [])}; decode ${JSON.stringify(decoder.outputs?.[0]?.shape || [])}`;
}

async function initialize() {
  try {
    const assetRootResolved = await resolveAssetRoot();
    if (!assetRootResolved) {
      setOnnxSizeText("ONNX Size: unavailable");
      setParameterText(null);
      weightsBadge.textContent = "Weights: missing";
      providerBadge.textContent = "Provider: unavailable";
      modelStatus.textContent = "Model: unavailable (missing demo assets)";
      ioStatus.textContent = "I/O: unavailable";
      outputStatus.textContent = "Output: unavailable";
      statsStatus.textContent = "Stats: unavailable";
      showError(missingAssetsMessage());
      return;
    }

    setOnnxSizeText("ONNX Size: probing...");
    const onnxSizeProbe = updateOnnxSizeStatus();
    modelStatus.textContent = "Runtime: loading ONNX Runtime Web...";
    ort = await loadOrtModule();
    modelStatus.textContent = "Model: loading metadata...";
    modelMeta = await loadJson(META_PATH);
    await onnxSizeProbe;
    setParameterText(modelMeta?.parameters?.total ?? modelMeta?.parameter_count);
    updateStatuses();

    try {
      promptPresets = await loadJson(PRESETS_PATH);
    } catch (_error) {
      promptPresets = { default: "a red cube on a table" };
    }

    try {
      schedulerConfig = await loadJson(SCHEDULER_CONFIG_PATH);
    } catch (_error) {
      schedulerConfig = {};
    }

    installPresets();
    const firstPreset = Object.keys(promptPresets)[0];
    if (firstPreset) {
      applyPreset(firstPreset);
    }

    const stepLimits = stepsRange();
    const guidanceLimits = guidanceRange();

    stepsInput.min = String(stepLimits.min);
    stepsInput.max = String(stepLimits.max);
    guidanceInput.min = guidanceLimits.min.toFixed(2);
    guidanceInput.max = guidanceLimits.max.toFixed(2);

    const defaultSteps = Math.max(
      stepLimits.min,
      Math.min(stepLimits.max, Math.trunc(toNumber(modelMeta?.generation?.default_steps, 20)))
    );
    stepsInput.value = String(defaultSteps);
    const defaultGuidance = clamp(
      toNumber(modelMeta?.generation?.default_guidance_scale, 7.5),
      guidanceLimits.min,
      guidanceLimits.max
    );
    guidanceInput.value = defaultGuidance.toFixed(2);

    const maxT = Math.max(1, Math.round(maxTime()));
    timestepInput.max = String(maxT);
    timestepInput.value = String(maxT);

    setStepsLabel();
    setGuidanceLabel();
    setTimestepLabel();

    const [hasTextEncoder, hasUnet, hasVaeDecoder] = await Promise.all([
      assetExists(TEXT_ENCODER_PATH),
      assetExists(UNET_PATH),
      assetExists(VAE_DECODER_PATH)
    ]);

    if (!(hasTextEncoder && hasUnet && hasVaeDecoder)) {
      weightsBadge.textContent = "Weights: missing";
      providerBadge.textContent = "Provider: unavailable";
      modelStatus.textContent = "Model: unavailable (missing ONNX assets)";
      return;
    }

    await loadTokenizer();

    if (modelMeta?.weights?.trained) {
      weightsBadge.textContent = `Weights: ${modelMeta?.weights?.repo_id || modelMeta?.weights?.source || "trained"}`;
    } else {
      weightsBadge.textContent = "Weights: random-init fallback";
    }

    const created = await createPipelineSessions();
    pipelineSessions = created.sessions;
    selectedProviders = created.providers;
    providerBadge.textContent = `Provider: text/unet=${selectedProviders.textUnet}, vae=${selectedProviders.vaeDecoder}`;
    setRunEnabled(true);
  } catch (error) {
    showError(String(error));
    setParameterText(null);
  }
}

presetSelect.addEventListener("change", () => applyPreset(presetSelect.value));
timestepInput.addEventListener("input", setTimestepLabel);
stepsInput.addEventListener("input", setStepsLabel);
stepsInput.addEventListener("change", setStepsLabel);
guidanceInput.addEventListener("input", setGuidanceLabel);
runButton.addEventListener("click", () => {
  runMultiStepDenoise();
});
newSeedButton.addEventListener("click", () => {
  randomizeSeed();
});

setStepsLabel();
setGuidanceLabel();
setTimestepLabel();
initialize();
