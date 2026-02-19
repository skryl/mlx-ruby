#!/usr/bin/env node

import http from "node:http";
import path from "node:path";
import process from "node:process";
import { promises as fs } from "node:fs";
import { chromium } from "playwright";

const REQUIRED_FILES = [
  "index.html",
  "harness.js",
  "harness.manifest.json",
  "inputs.example.json",
  "model.onnx"
];

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".data": "application/octet-stream"
};

const MOCK_ORT_MODULE = `
export class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

class MockSession {
  constructor(provider) {
    this.provider = provider;
    this.runCount = 0;
  }

  async run(_feeds) {
    this.runCount += 1;
    return {};
  }
}

export const InferenceSession = {
  async create(_modelPath, options = {}) {
    const providers = options.executionProviders || [];
    const provider = providers[0] || "wasm";
    if (provider === "webgpu") {
      throw new Error("mock webgpu provider unavailable");
    }
    return new MockSession(provider);
  }
};
`;

function parseArgs(argv) {
  const options = {
    harnessDir: null,
    timeoutSeconds: 30,
    mockOrt: false,
    localOrt: true
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--harness-dir") {
      options.harnessDir = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--timeout-seconds") {
      options.timeoutSeconds = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === "--mock-ort") {
      options.mockOrt = true;
      continue;
    }
    if (arg === "--local-ort") {
      options.localOrt = true;
      continue;
    }
    if (arg === "--no-local-ort") {
      options.localOrt = false;
      continue;
    }
    throw new Error(`unsupported argument: ${arg}`);
  }

  if (!options.harnessDir) {
    throw new Error("missing required --harness-dir argument");
  }
  if (!Number.isFinite(options.timeoutSeconds) || options.timeoutSeconds <= 0) {
    throw new Error("--timeout-seconds must be a positive number");
  }

  return options;
}

async function resolveLocalOrtDistDir() {
  const distDir = path.resolve(process.cwd(), "node_modules", "onnxruntime-web", "dist");
  const moduleEntry = path.join(distDir, "ort.all.min.mjs");
  try {
    const stat = await fs.stat(moduleEntry);
    if (!stat.isFile()) {
      throw new Error(`onnxruntime-web entry is not a file: ${moduleEntry}`);
    }
  } catch (error) {
    if (error && error.code === "ENOENT") {
      throw new Error(
        `missing onnxruntime-web distribution under ${distDir}; run 'npm install' in the web/ directory`
      );
    }
    throw error;
  }
  return distDir;
}

function contentTypeForPath(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return CONTENT_TYPES[ext] || "application/octet-stream";
}

async function ensureHarnessFiles(harnessDir) {
  for (const fileName of REQUIRED_FILES) {
    const candidate = path.join(harnessDir, fileName);
    try {
      const stat = await fs.stat(candidate);
      if (!stat.isFile()) {
        throw new Error(`required harness file is not a file: ${candidate}`);
      }
    } catch (error) {
      if (error && error.code === "ENOENT") {
        throw new Error(`missing required harness file: ${candidate}`);
      }
      throw error;
    }
  }
}

async function startStaticServer(rootDir) {
  const absoluteRoot = path.resolve(rootDir);
  const server = http.createServer(async (req, res) => {
    const requestUrl = new URL(req.url || "/", "http://127.0.0.1");
    let pathname = decodeURIComponent(requestUrl.pathname);
    if (pathname === "/") {
      pathname = "/index.html";
    }

    const relPath = pathname.replace(/^\/+/, "");
    const candidate = path.resolve(absoluteRoot, relPath);
    if (!(candidate === absoluteRoot || candidate.startsWith(`${absoluteRoot}${path.sep}`))) {
      res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
      res.end("forbidden");
      return;
    }

    try {
      const stat = await fs.stat(candidate);
      if (!stat.isFile()) {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end("not found");
        return;
      }

      const body = await fs.readFile(candidate);
      res.writeHead(200, {
        "cache-control": "no-store",
        "content-type": contentTypeForPath(candidate)
      });
      res.end(body);
    } catch (error) {
      if (error && error.code === "ENOENT") {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end("not found");
        return;
      }
      res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      res.end(`internal server error: ${String(error)}`);
    }
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("failed to bind local HTTP server");
  }

  return {
    server,
    rootUrl: `http://127.0.0.1:${address.port}/`
  };
}

async function runSmoke(options) {
  const timeoutMs = Math.ceil(options.timeoutSeconds * 1000);
  const harnessDir = path.resolve(options.harnessDir);
  await ensureHarnessFiles(harnessDir);

  const { server, rootUrl } = await startStaticServer(harnessDir);
  let browser;
  let context;

  try {
    browser = await chromium.launch({
      headless: true,
      args: ["--enable-unsafe-webgpu"]
    });
    context = await browser.newContext();

    const pageErrors = [];
    const consoleErrors = [];
    const page = await context.newPage();
    page.on("pageerror", (error) => {
      pageErrors.push(String(error));
    });
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text());
      }
    });

    if (options.mockOrt) {
      await context.route("**/onnxruntime-web/dist/ort.all.min.mjs", async (route) => {
        await route.fulfill({
          status: 200,
          body: MOCK_ORT_MODULE,
          contentType: "text/javascript; charset=utf-8",
          headers: {
            "access-control-allow-origin": "*",
            "cache-control": "no-store"
          }
        });
      });
    } else if (options.localOrt) {
      const localOrtDistDir = await resolveLocalOrtDistDir();
      await context.route("**/onnxruntime-web/dist/**", async (route) => {
        const url = new URL(route.request().url());
        const marker = "/onnxruntime-web/dist/";
        const markerIndex = url.pathname.indexOf(marker);
        if (markerIndex < 0) {
          await route.continue();
          return;
        }

        const relPath = decodeURIComponent(url.pathname.slice(markerIndex + marker.length));
        const candidate = path.resolve(localOrtDistDir, relPath);
        const rootPrefix = `${localOrtDistDir}${path.sep}`;
        if (!(candidate === localOrtDistDir || candidate.startsWith(rootPrefix))) {
          await route.fulfill({
            status: 403,
            body: "forbidden",
            contentType: "text/plain; charset=utf-8"
          });
          return;
        }

        try {
          const stat = await fs.stat(candidate);
          if (!stat.isFile()) {
            await route.fulfill({
              status: 404,
              body: "not found",
              contentType: "text/plain; charset=utf-8"
            });
            return;
          }

          const body = await fs.readFile(candidate);
          await route.fulfill({
            status: 200,
            body,
            contentType: contentTypeForPath(candidate),
            headers: {
              "access-control-allow-origin": "*",
              "cache-control": "no-store"
            }
          });
        } catch (error) {
          if (error && error.code === "ENOENT") {
            await route.fulfill({
              status: 404,
              body: "not found",
              contentType: "text/plain; charset=utf-8"
            });
            return;
          }
          await route.fulfill({
            status: 500,
            body: `failed to serve local onnxruntime-web asset: ${String(error)}`,
            contentType: "text/plain; charset=utf-8"
          });
        }
      });
    }

    await page.goto(new URL("index.html", rootUrl).toString(), {
      waitUntil: "networkidle",
      timeout: timeoutMs
    });
    await page.waitForSelector("#run-button", { timeout: timeoutMs });
    await page.waitForFunction(
      () => {
        const text = document.getElementById("status")?.textContent || "";
        return text.includes("Ready.");
      },
      { timeout: timeoutMs }
    );

    await page.click("#run-button");
    await page.waitForFunction(
      () => {
        const text = document.getElementById("status")?.textContent || "";
        try {
          const payload = JSON.parse(text);
          return payload && (payload.format === "onnx_webgpu_telemetry_v1" || typeof payload.error === "string");
        } catch (_error) {
          return false;
        }
      },
      { timeout: timeoutMs }
    );

    const statusText = await page.$eval("#status", (node) => node.textContent || "");
    let telemetry;
    try {
      telemetry = JSON.parse(statusText);
    } catch (error) {
      throw new Error(`harness status is not valid JSON: ${String(error)}`);
    }

    if (telemetry && typeof telemetry.error === "string") {
      throw new Error(`harness run failed: ${telemetry.error}`);
    }
    if (!telemetry || telemetry.format !== "onnx_webgpu_telemetry_v1") {
      throw new Error(`unexpected telemetry payload: ${statusText}`);
    }

    telemetry.page_errors = pageErrors;
    telemetry.console_errors = consoleErrors;
    return telemetry;
  } finally {
    if (context) {
      await context.close().catch(() => {});
    }
    if (browser) {
      await browser.close().catch(() => {});
    }
    await new Promise((resolve) => server.close(resolve));
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const telemetry = await runSmoke(options);
  process.stdout.write(`${JSON.stringify(telemetry)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
