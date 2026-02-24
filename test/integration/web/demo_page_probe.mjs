#!/usr/bin/env node

import process from "node:process";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PLAYWRIGHT_ENTRY = path.join(SCRIPT_DIR, "..", "..", "..", "web", "node_modules", "playwright", "index.mjs");
const { chromium } = await import(pathToFileURL(PLAYWRIGHT_ENTRY).href);

const DEMO_PATHS = {
  gpt2: "/demo/gpt2/",
  nanogpt: "/demo/nanogpt/",
  stable_diffusion: "/demo/stable_diffusion/"
};

const EXPECTED_MODEL_NAMES = {
  gpt2: "gpt2_ruby_web_demo",
  nanogpt: "nanogpt_shakespeare_web_demo",
  stable_diffusion: "stable_diffusion_kanji_web_demo"
};

function parseArgs(argv) {
  const options = { baseUrl: null, demo: null };
  for (let idx = 0; idx < argv.length; idx += 1) {
    const arg = argv[idx];
    if (arg === "--base-url") {
      options.baseUrl = argv[idx + 1] || null;
      idx += 1;
      continue;
    }
    if (arg === "--demo") {
      options.demo = argv[idx + 1] || null;
      idx += 1;
      continue;
    }
    throw new Error(`unsupported argument: ${arg}`);
  }

  if (!options.baseUrl) {
    throw new Error("missing --base-url");
  }
  if (!options.demo || !Object.prototype.hasOwnProperty.call(DEMO_PATHS, options.demo)) {
    throw new Error(`missing or unsupported --demo (expected one of: ${Object.keys(DEMO_PATHS).join(", ")})`);
  }
  return options;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function routeStableDiffusionMetaAsTrained(context) {
  await context.route("**/assets/stable_diffusion/meta.json", async (route) => {
    const response = await route.fetch();
    const source = await response.text();
    let payload;
    try {
      payload = JSON.parse(source);
    } catch (_error) {
      await route.fulfill({ response });
      return;
    }

    payload.weights = payload.weights || {};
    payload.weights.trained = true;

    const headers = {
      ...response.headers(),
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store"
    };

    await route.fulfill({
      response,
      headers,
      body: JSON.stringify(payload)
    });
  });
}

async function waitForReadyState(page, demo) {
  const buttonId = demo === "stable_diffusion" ? "run" : "generate";
  await page.waitForFunction(
    (id) => {
      const model = document.getElementById("status-model")?.textContent || "";
      const onnxSize = document.getElementById("status-onnx-size")?.textContent || "";
      const provider = document.getElementById("badge-provider")?.textContent || "";
      const parameters = document.getElementById("badge-parameters")?.textContent || "";
      const button = document.getElementById(id);
      return (
        Boolean(button) &&
        button.disabled === false &&
        model.startsWith("Model:") &&
        !model.includes("loading") &&
        !model.includes("failed") &&
        !model.includes("unavailable") &&
        onnxSize.startsWith("ONNX Size:") &&
        !onnxSize.includes("loading") &&
        provider.startsWith("Provider:") &&
        parameters.startsWith("Parameters:") &&
        !parameters.includes("loading")
      );
    },
    buttonId,
    { timeout: 180000 }
  );
}

async function collectStatus(page, demo) {
  const buttonId = demo === "stable_diffusion" ? "run" : "generate";
  return page.evaluate(({ id, mode }) => {
    const text = (selector) => document.querySelector(selector)?.textContent?.trim() || "";
    const button = document.getElementById(id);
    const topKRows = document.querySelectorAll("#topk-body tr").length;

    return {
      model_status: text("#status-model"),
      provider_badge: text("#badge-provider"),
      weights_badge: text("#badge-weights"),
      onnx_size_badge: text("#badge-onnx-size"),
      parameters_badge: text("#badge-parameters"),
      onnx_size_status: text("#status-onnx-size"),
      timing_badge: text("#badge-timing"),
      tokenizer_status: text("#status-tokenizer"),
      context_status: text("#status-context"),
      output_status: text("#status-output"),
      stats_status: text("#status-stats"),
      output_text: text("#output-text"),
      output_ids: text("#output-ids"),
      preview_values: text("#preview-values"),
      topk_rows: topKRows,
      generate_enabled: Boolean(button) && button.disabled === false,
      demo: mode
    };
  }, { id: buttonId, mode: demo });
}

async function runGeneration(page, demo) {
  if (demo === "stable_diffusion") {
    await page.fill("#steps", "1");
    await page.evaluate(() => {
      const guidance = document.getElementById("guidance");
      if (guidance) {
        guidance.value = "0";
        guidance.dispatchEvent(new Event("input", { bubbles: true }));
      }
      const timestep = document.getElementById("timestep");
      if (timestep) {
        timestep.value = "1";
        timestep.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    await page.click("#run");
    await page.waitForFunction(
      () => {
        const timing = document.getElementById("badge-timing")?.textContent || "";
        const output = document.getElementById("status-output")?.textContent || "";
        const stats = document.getElementById("status-stats")?.textContent || "";
        return (
          timing.startsWith("Inference:") &&
          !timing.includes("--") &&
          output.startsWith("Output:") &&
          !output.endsWith("--") &&
          stats.startsWith("Stats:") &&
          !stats.endsWith("--")
        );
      },
      null,
      { timeout: 240000 }
    );
    return;
  }

  await page.fill("#max-tokens", "1");
  await page.evaluate(() => {
    const temperature = document.getElementById("temperature");
    if (temperature) {
      temperature.value = "0";
      temperature.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  await page.click("#generate");
  await page.waitForFunction(
    () => {
      const timing = document.getElementById("badge-timing")?.textContent || "";
      const outputText = document.getElementById("output-text")?.textContent || "";
      const outputIds = document.getElementById("output-ids")?.textContent || "";
      const topKRows = document.querySelectorAll("#topk-body tr").length;
      return (
        timing.startsWith("Inference:") &&
        !timing.includes("--") &&
        (
          outputText.trim().length > 0 ||
          outputIds.trim().length > 0 ||
          topKRows > 0
        )
      );
    },
    null,
    { timeout: 180000 }
  );
}

function validateResult(result) {
  const demo = result.demo;
  const expectedModelName = EXPECTED_MODEL_NAMES[demo];
  const pre = result.pre;
  const post = result.post;

  assert(pre.model_status.includes(expectedModelName), `${demo}: expected model status to include ${expectedModelName}, got: ${pre.model_status}`);
  assert(pre.provider_badge.startsWith("Provider:"), `${demo}: expected provider badge, got: ${pre.provider_badge}`);
  assert(pre.weights_badge.startsWith("Weights:"), `${demo}: expected weights badge, got: ${pre.weights_badge}`);
  assert(pre.onnx_size_badge.startsWith("ONNX Size:"), `${demo}: expected ONNX size badge, got: ${pre.onnx_size_badge}`);
  assert(!pre.onnx_size_badge.includes("loading"), `${demo}: ONNX size badge still loading`);
  assert(pre.parameters_badge.startsWith("Parameters:"), `${demo}: expected parameter badge, got: ${pre.parameters_badge}`);
  assert(!pre.parameters_badge.includes("loading"), `${demo}: parameter badge still loading`);
  assert(
    /^Parameters:\s[0-9][0-9,]*$/u.test(pre.parameters_badge),
    `${demo}: expected formatted parameter count, got: ${pre.parameters_badge}`
  );
  assert(pre.onnx_size_status.startsWith("ONNX Size:"), `${demo}: expected ONNX size status, got: ${pre.onnx_size_status}`);
  assert(!pre.onnx_size_status.includes("loading"), `${demo}: ONNX size status still loading`);
  assert(pre.generate_enabled === true, `${demo}: expected action button to be enabled`);

  assert(post.timing_badge.startsWith("Inference:"), `${demo}: expected inference timing, got: ${post.timing_badge}`);
  assert(!post.timing_badge.includes("--"), `${demo}: inference timing still unset`);

  if (demo === "stable_diffusion") {
    assert(post.output_status.startsWith("Output:"), `${demo}: expected output status, got: ${post.output_status}`);
    assert(post.stats_status.startsWith("Stats:"), `${demo}: expected stats status, got: ${post.stats_status}`);
    assert(post.preview_values.length > 0, `${demo}: expected preview values`);
    return;
  }

  const hasOutputText = post.output_text.length > 0;
  const hasOutputIds = post.output_ids.length > 0 && post.output_ids !== "[]";
  assert(hasOutputText || hasOutputIds, `${demo}: expected generated output text or token ids`);
  assert(post.topk_rows > 0, `${demo}: expected top-k rows`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const demoUrl = new URL(DEMO_PATHS[options.demo], options.baseUrl).toString();
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-unsafe-webgpu"]
  });
  const context = await browser.newContext();
  const pageErrors = [];
  const consoleErrors = [];

  try {
    if (options.demo === "stable_diffusion") {
      await routeStableDiffusionMetaAsTrained(context);
    }

    const page = await context.newPage();
    page.on("pageerror", (error) => {
      pageErrors.push(String(error));
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto(demoUrl, { waitUntil: "domcontentloaded", timeout: 120000 });
    await waitForReadyState(page, options.demo);
    const pre = await collectStatus(page, options.demo);
    await runGeneration(page, options.demo);
    const post = await collectStatus(page, options.demo);

    const result = {
      demo: options.demo,
      url: demoUrl,
      pre,
      post,
      console_errors: consoleErrors,
      page_errors: pageErrors
    };

    validateResult(result);
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error && error.stack ? error.stack : error)}\n`);
  process.exit(1);
});
