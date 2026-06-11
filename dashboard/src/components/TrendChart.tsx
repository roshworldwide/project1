/** Metric-over-time trend: shaded CI band, thin gold line, glass tooltip. */

import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { RunSummary } from "../api";
import { fmtCI, fmtDate, fmtDateShort, fmtPct, niceStep, ticksIn } from "../format";
import { useMeasure } from "../hooks";

interface TrendChartProps {
  runs: RunSummary[]; // ascending by created_at
  metric: string;
  onPick?: (run: RunSummary) => void;
}

const HEIGHT = 240;
const M = { top: 14, right: 18, bottom: 30, left: 48 };

export function TrendChart({ runs, metric, onPick }: TrendChartProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const pts = useMemo(
    () =>
      runs
        .filter((r) => r.metrics[metric])
        .map((r) => ({
          run: r,
          t: new Date(r.created_at).getTime(),
          est: r.metrics[metric]!,
        })),
    [runs, metric],
  );

  const geom = useMemo(() => {
    if (width <= 0 || pts.length === 0) return null;
    const innerW = width - M.left - M.right;
    const innerH = HEIGHT - M.top - M.bottom;
    const t0 = Math.min(...pts.map((p) => p.t));
    const t1 = Math.max(...pts.map((p) => p.t));
    const tSpan = Math.max(1, t1 - t0);
    let lo = Math.min(...pts.map((p) => p.est.ci_low));
    let hi = Math.max(...pts.map((p) => p.est.ci_high));
    const padY = Math.max(0.015, (hi - lo) * 0.18);
    lo = Math.max(0, lo - padY);
    hi = Math.min(1, hi + padY);
    const x = (t: number) => M.left + ((t - t0) / tSpan) * innerW;
    const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * innerH;
    const step = niceStep(hi - lo);
    return { x, y, lo, hi, yTicks: ticksIn(lo, hi, step) };
  }, [width, pts]);

  if (pts.length === 0) return null;

  let band = "";
  let line = "";
  let edgeHigh = "";
  let edgeLow = "";
  if (geom) {
    const top = pts.map(
      (p, i) => `${i ? "L" : "M"}${geom.x(p.t)},${geom.y(p.est.ci_high)}`,
    );
    const bottom = [...pts]
      .reverse()
      .map((p) => `L${geom.x(p.t)},${geom.y(p.est.ci_low)}`);
    band = `${top.join(" ")} ${bottom.join(" ")} Z`;
    edgeHigh = top.join(" ");
    edgeLow = pts
      .map((p, i) => `${i ? "L" : "M"}${geom.x(p.t)},${geom.y(p.est.ci_low)}`)
      .join(" ");
    line = pts
      .map((p, i) => `${i ? "L" : "M"}${geom.x(p.t)},${geom.y(p.est.value)}`)
      .join(" ");
  }

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!geom) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    let best = 0;
    let bestD = Infinity;
    pts.forEach((p, i) => {
      const d = Math.abs(geom.x(p.t) - px);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    setHover(best);
  };

  const h = hover !== null && geom ? pts[hover] : null;

  return (
    <div ref={ref} style={{ position: "relative", width: "100%" }}>
      {geom && (
        <svg
          width={width}
          height={HEIGHT}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onClick={() => h && onPick?.(h.run)}
          style={{ display: "block", cursor: onPick ? "pointer" : "default" }}
          aria-label={`${metric} over time`}
        >
          {/* muted grid + y labels */}
          {geom.yTicks.map((t) => (
            <g key={t}>
              <line
                x1={M.left}
                x2={width - M.right}
                y1={geom.y(t)}
                y2={geom.y(t)}
                stroke="var(--grid-line)"
                strokeWidth={1}
              />
              <text
                x={M.left - 10}
                y={geom.y(t) + 3.5}
                textAnchor="end"
                fontSize={10}
                fill="var(--text-3)"
                fontFamily="var(--font-mono)"
              >
                {fmtPct(t, 0)}
              </text>
            </g>
          ))}

          {/* x labels */}
          {pts.map((p, i) => (
            <text
              key={p.run.run_id}
              x={geom.x(p.t)}
              y={HEIGHT - 9}
              textAnchor={
                i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle"
              }
              fontSize={10}
              fill="var(--text-3)"
            >
              {fmtDateShort(p.run.created_at)}
            </text>
          ))}

          {/* CI band with faint glassy edges */}
          <motion.path
            d={band}
            fill="var(--gold-data)"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.13 }}
            transition={{ duration: 0.9, delay: 0.15 }}
          />
          {[edgeHigh, edgeLow].map((d, i) => (
            <motion.path
              key={i}
              d={d}
              fill="none"
              stroke="var(--gold-data)"
              strokeWidth={1}
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.22 }}
              transition={{ duration: 0.9, delay: 0.3 }}
            />
          ))}

          {/* crosshair */}
          {h && (
            <line
              x1={geom.x(h.t)}
              x2={geom.x(h.t)}
              y1={M.top}
              y2={HEIGHT - M.bottom}
              stroke="var(--hairline)"
              strokeWidth={1}
            />
          )}

          {/* gold line */}
          <motion.path
            d={line}
            fill="none"
            stroke="var(--gold-data)"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ type: "spring", duration: 1.3, bounce: 0 }}
          />

          {/* hover CI whisker */}
          {h && (
            <g stroke="var(--gold-data)" strokeWidth={1.5} strokeLinecap="round" opacity={0.6}>
              <line
                x1={geom.x(h.t)}
                x2={geom.x(h.t)}
                y1={geom.y(h.est.ci_low)}
                y2={geom.y(h.est.ci_high)}
              />
              <line x1={geom.x(h.t) - 3.5} x2={geom.x(h.t) + 3.5} y1={geom.y(h.est.ci_low)} y2={geom.y(h.est.ci_low)} />
              <line x1={geom.x(h.t) - 3.5} x2={geom.x(h.t) + 3.5} y1={geom.y(h.est.ci_high)} y2={geom.y(h.est.ci_high)} />
            </g>
          )}

          {/* points */}
          {pts.map((p, i) => (
            <motion.circle
              key={p.run.run_id}
              cx={geom.x(p.t)}
              cy={geom.y(p.est.value)}
              r={hover === i ? 4.5 : 3}
              fill="var(--gold-data)"
              stroke="var(--canvas)"
              strokeWidth={1.5}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{
                type: "spring",
                stiffness: 320,
                damping: 20,
                delay: 0.25 + i * 0.05,
              }}
              style={{ transformOrigin: `${geom.x(p.t)}px ${geom.y(p.est.value)}px` }}
            />
          ))}
        </svg>
      )}

      {/* glass tooltip */}
      <AnimatePresence>
        {h && geom && (
          <motion.div
            key="tip"
            className="glass-tooltip"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{
              opacity: 1,
              scale: 1,
              left: Math.min(Math.max(geom.x(h.t), 90), width - 90),
              top: geom.y(h.est.value),
            }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 380, damping: 28 }}
            style={{ transform: "translate(-50%, -120%)", translateX: "-50%", translateY: "-122%" }}
          >
            <div className="t-title">{h.run.target_name}</div>
            <div className="t-sub mono">
              {h.run.short_run_id} · {fmtDate(h.run.created_at)}
            </div>
            <div className="t-val">
              <strong>{fmtPct(h.est.value)}</strong>{" "}
              <span style={{ color: "var(--text-3)" }}>
                CI {fmtCI(h.est.ci_low, h.est.ci_high)}
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
