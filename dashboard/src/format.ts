/** Number/date formatting helpers, tuned for stats honesty + Apple polish. */

export function fmtPct(v: number, decimals = 1): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtCI(low: number, high: number): string {
  return `${(low * 100).toFixed(1)}–${(high * 100).toFixed(1)}%`;
}

/** Signed effect in percentage points: “−10.8 pp”. */
export function fmtEffect(v: number): string {
  const pts = v * 100;
  const sign = pts > 0 ? "+" : pts < 0 ? "−" : "±";
  return `${sign}${Math.abs(pts).toFixed(1)} pp`;
}

export function fmtP(p: number): string {
  if (p < 0.001) return "p < 0.001";
  return `p = ${p.toFixed(3)}`;
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function fmtDateShort(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function fmtLatency(s: number | null): string {
  if (s === null) return "—";
  if (s < 0.001) return `${(s * 1e6).toFixed(0)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(1)} ms`;
  return `${s.toFixed(2)} s`;
}

/** A nicely-rounded tick step for an axis spanning `range`. */
export function niceStep(range: number, targetTicks = 4): number {
  const raw = range / Math.max(1, targetTicks);
  const mag = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (raw <= m * mag) return m * mag;
  }
  return 10 * mag;
}

export function ticksIn(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  let t = Math.ceil(lo / step) * step;
  for (; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(10)));
  return out;
}
