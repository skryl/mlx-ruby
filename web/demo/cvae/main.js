import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs";

const ASSET_ROOT = "../../assets/cvae";
const MODEL_PATH = `${ASSET_ROOT}/model.onnx`;
const META_PATH = `${ASSET_ROOT}/meta.json`;
const PRESETS_PATH = `${ASSET_ROOT}/input.presets.json`;

const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d", { alpha: false });
ctx.imageSmoothingEnabled = false;

const weightsBadge = document.getElementById("badge-weights");
const providerBadge = document.getElementById("badge-provider");
const timingBadge = document.getElementById("badge-timing");
const modelStatus = document.getElementById("status-model");
const shapeStatus = document.getElementById("status-shape");
const errorBox = document.getElementById("error-box");

const presetSelect = document.getElementById("preset");
const presetName = document.getElementById("preset-name");
const randomizeButton = document.getElementById("randomize");
const resetButton = document.getElementById("reset");

const sliderIds = ["z0", "z1", "z2", "z3"];
const sliders = sliderIds.map((id) => document.getElementById(id));
const sliderValueLabels = sliderIds.map((id) => document.getElementById(`${id}-value`));

let session = null;
let selectedProvider = null;
let modelMeta = null;
let presets = {};
let runToken = 0;
let runTimer = null;

setControlsEnabled(false);

function showError(message) {
  errorBox.style.display = "block";
  errorBox.textContent = message;
}

function clearError() {
  errorBox.style.display = "none";
  errorBox.textContent = "";
}

function clampUnit(value) {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function updateSliderReadouts() {
  sliders.forEach((slider, index) => {
    sliderValueLabels[index].textContent = Number(slider.value).toFixed(2);
  });
}

function currentLatentVector() {
  return sliders.map((slider) => Number(slider.value));
}

function setControlsEnabled(enabled) {
  const disabled = !enabled;
  presetSelect.disabled = disabled;
  randomizeButton.disabled = disabled;
  resetButton.disabled = disabled;
  sliders.forEach((slider) => {
    slider.disabled = disabled;
  });
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

function applyLatentVector(values) {
  sliders.forEach((slider, index) => {
    slider.value = String(values[index] ?? 0);
  });
  updateSliderReadouts();
}

function applyPreset(name) {
  if (!(name in presets)) {
    return;
  }
  presetSelect.value = name;
  presetName.textContent = name;
  applyLatentVector(presets[name]);
  scheduleInference();
}

function randomLatent() {
  return Array.from({ length: 4 }, () => (Math.random() * 6 - 3));
}

function createFeeds(vector) {
  const inputSpec = modelMeta.input;
  const data = Float32Array.from(vector);
  return {
    [inputSpec.name]: new ort.Tensor("float32", data, inputSpec.shape)
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

function renderGrayscaleImage(tensor) {
  const dims = tensor.dims || [];
  if (dims.length !== 4 || dims[0] !== 1 || dims[3] !== 1) {
    throw new Error(`unexpected output shape: ${JSON.stringify(dims)}`);
  }
  const height = dims[1];
  const width = dims[2];
  const data = tensor.data;
  if (!data || data.length < width * height) {
    throw new Error("invalid output tensor data");
  }

  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const value = Math.round(clampUnit(Number(data[i])) * 255);
    const pixel = i * 4;
    imageData.data[pixel + 0] = value;
    imageData.data[pixel + 1] = value;
    imageData.data[pixel + 2] = value;
    imageData.data[pixel + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

async function runInferenceNow() {
  if (!session || !modelMeta) {
    return;
  }

  clearError();
  const token = ++runToken;
  const feeds = createFeeds(currentLatentVector());
  const started = performance.now();
  const outputs = await session.run(feeds);
  const elapsedMs = performance.now() - started;
  if (token !== runToken) {
    return;
  }

  const tensor = outputTensor(outputs);
  renderGrayscaleImage(tensor);
  timingBadge.textContent = `Inference: ${elapsedMs.toFixed(2)} ms`;
  shapeStatus.textContent = `Output shape: ${JSON.stringify(tensor.dims)}`;
}

function scheduleInference(delayMs = 30) {
  if (runTimer) {
    clearTimeout(runTimer);
  }
  runTimer = setTimeout(() => {
    runTimer = null;
    runInferenceNow().catch((error) => {
      showError(String(error));
    });
  }, delayMs);
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`failed to fetch ${path}: HTTP ${response.status}`);
  }
  return response.json();
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

function installUiHandlers() {
  sliders.forEach((slider) => {
    slider.addEventListener("input", () => {
      updateSliderReadouts();
      scheduleInference();
    });
  });

  presetSelect.addEventListener("change", () => {
    applyPreset(presetSelect.value);
  });

  randomizeButton.addEventListener("click", () => {
    applyLatentVector(randomLatent());
    scheduleInference();
  });

  resetButton.addEventListener("click", () => {
    applyPreset("calm");
  });
}

function installPresets() {
  presetSelect.innerHTML = "";
  Object.keys(presets).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    presetSelect.appendChild(option);
  });
}

async function boot() {
  try {
    setControlsEnabled(false);
    modelStatus.textContent = "Model: loading metadata...";
    [modelMeta, presets] = await Promise.all([loadJson(META_PATH), loadJson(PRESETS_PATH)]);
    installPresets();
    installUiHandlers();
    updateSliderReadouts();

    const hasModelAsset = await assetExists(MODEL_PATH);
    if (!hasModelAsset) {
      weightsBadge.textContent = "Weights: missing (run `bundle exec rake web:assets`)";
      providerBadge.textContent = "Provider: unavailable";
      modelStatus.textContent = "Model: unavailable (missing weights)";
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

    weightsBadge.textContent = "Weights: embedded ONNX";
    modelStatus.textContent = `Model: ${modelMeta.model_name}`;
    setControlsEnabled(true);
    applyPreset("calm");
  } catch (error) {
    showError(String(error));
    weightsBadge.textContent = "Weights: missing (unavailable)";
    providerBadge.textContent = "Provider: unavailable";
    modelStatus.textContent = "Model: failed to load";
    shapeStatus.textContent = "Output shape: unavailable";
    setControlsEnabled(false);
  }
}

boot();
