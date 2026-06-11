/**
 * Capture launch screenshots of the dashboard served by the Python server.
 *
 * Prereq: the server is running on http://127.0.0.1:4517 with built assets.
 *   cd ".." && .venv/bin/holdout --store .holdout-demo dashboard --port 4517 --no-open
 *
 * Usage: node scripts/screenshot.mjs
 */

import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const BASE = process.env.DASHBOARD_URL ?? "http://127.0.0.1:4517";
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

// The demo pair that genuinely regresses: support-qa prompt-v5 → prompt-v4.
async function regressedPair() {
  const res = await fetch(`${BASE}/api/runs`);
  const { runs } = await res.json();
  const byTarget = (t) =>
    runs.find((r) => r.eval_name === "support-qa" && r.target_name === t);
  const baseline = byTarget("prompt-v5") ?? runs[0];
  const candidate = byTarget("prompt-v4") ?? runs[1];
  return `/compare?baseline=${baseline.run_id}&candidate=${candidate.run_id}`;
}

const SHOTS = [
  {
    path: "/?eval=support-qa",
    theme: "dark",
    out: "docs/assets/dashboard-dark.png",
  },
  {
    path: "/?eval=support-qa",
    theme: "light",
    out: "docs/assets/dashboard-light.png",
  },
  {
    path: await regressedPair(),
    theme: "dark",
    out: "docs/assets/dashboard-compare-dark.png",
  },
];

const browser = await chromium.launch();
for (const shot of SHOTS) {
  const ctx = await browser.newContext({
    viewport: { width: 1640, height: 1024 },
    deviceScaleFactor: 2,
    colorScheme: shot.theme,
  });
  const page = await ctx.newPage();
  // The app stores an explicit theme choice; clear it so colorScheme rules.
  await page.addInitScript(() => localStorage.removeItem("holdout-theme"));
  await page.goto(BASE + shot.path, { waitUntil: "networkidle" });
  // Let springs, count-ups and draw-ins settle.
  await page.waitForTimeout(3200);
  const out = resolve(ROOT, shot.out);
  mkdirSync(dirname(out), { recursive: true });
  await page.screenshot({ path: out });
  console.log(`captured ${shot.out} (${shot.theme})`);
  await ctx.close();
}
await browser.close();
