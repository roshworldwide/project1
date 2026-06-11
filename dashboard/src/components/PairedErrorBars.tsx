/** The signature visual: baseline vs candidate CIs on one shared axis.
 *  Bars spring outward from their point estimate on mount. */

import { motion } from "framer-motion";
import { useMemo } from "react";
import type { Estimate } from "../api";
import { fmtPct, niceStep, ticksIn } from "../format";
import { useMeasure } from "../hooks";

interface PairedErrorBarsProps {
  baseline: Estimate;
  candidate: Estimate;
  baselineLabel: string;
  candidateLabel: string;
}

const HEIGHT = 138;
const LABEL_W = 150;
const M = { right: 64, top: 16 };
const ROW_B = 42;
const ROW_C = 84;
const AXIS_Y = 112;

function Whisker({
  est,
  x,
  y,
  color,
  opacity,
  delay,
}: {
  est: Estimate;
  x: (v: number) => number;
  y: number;
  color: string;
  opacity: number;
  delay: number;
}) {
  const cx = x(est.value);
  const cap = 7;
  return (
    <g transform={`translate(${cx}, ${y})`}>
      <motion.g
        initial={{ scaleX: 0, opacity: 0 }}
        animate={{ scaleX: 1, opacity }}
        transition={{ type: "spring", stiffness: 110, damping: 19, delay }}
      >
        <line
          x1={x(est.ci_low) - cx}
          x2={x(est.ci_high) - cx}
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
        />
        <line
          x1={x(est.ci_low) - cx}
          x2={x(est.ci_low) - cx}
          y1={-cap}
          y2={cap}
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
        />
        <line
          x1={x(est.ci_high) - cx}
          x2={x(est.ci_high) - cx}
          y1={-cap}
          y2={cap}
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
        />
      </motion.g>
      <motion.circle
        r={5}
        fill={color}
        stroke="var(--canvas)"
        strokeWidth={1.5}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 17, delay: delay + 0.12 }}
      />
    </g>
  );
}

export function PairedErrorBars({
  baseline,
  candidate,
  baselineLabel,
  candidateLabel,
}: PairedErrorBarsProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();

  const geom = useMemo(() => {
    if (width <= 0) return null;
    let lo = Math.min(baseline.ci_low, candidate.ci_low);
    let hi = Math.max(baseline.ci_high, candidate.ci_high);
    if (hi - lo < 0.02) {
      lo -= 0.03;
      hi += 0.03;
    }
    const pad = Math.max(0.012, (hi - lo) * 0.16);
    lo = Math.max(0, lo - pad);
    hi = Math.min(1, hi + pad);
    const x = (v: number) =>
      LABEL_W + ((v - lo) / (hi - lo)) * (width - LABEL_W - M.right);
    const step = niceStep(hi - lo, 5);
    return { x, lo, hi, ticks: ticksIn(lo, hi, step) };
  }, [width, baseline, candidate]);

  const valueLabel = (est: Estimate, y: number, color: string) => {
    if (!geom) return null;
    // Place after the high cap; flip to before the low cap near the edge.
    const fitsRight = geom.x(est.ci_high) + 58 < width;
    const xEnd = fitsRight ? geom.x(est.ci_high) + 12 : geom.x(est.ci_low) - 12;
    return (
      <motion.text
        x={xEnd}
        y={y + 4}
        fontSize={12}
        fontWeight={600}
        fill={color}
        fontFamily="var(--font-mono)"
        textAnchor={fitsRight ? "start" : "end"}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.4 }}
      >
        {fmtPct(est.value)}
      </motion.text>
    );
  };

  return (
    <div ref={ref} style={{ width: "100%" }}>
      {geom && (
        <svg width={width} height={HEIGHT} style={{ display: "block" }} aria-hidden>
          {/* vertical gridlines + axis tick labels */}
          {geom.ticks.map((t) => (
            <g key={t}>
              <line
                x1={geom.x(t)}
                x2={geom.x(t)}
                y1={M.top}
                y2={AXIS_Y - 8}
                stroke="var(--grid-line)"
                strokeWidth={1}
              />
              <text
                x={geom.x(t)}
                y={AXIS_Y + 8}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-3)"
                fontFamily="var(--font-mono)"
              >
                {fmtPct(t, 0)}
              </text>
            </g>
          ))}

          {/* dashed reference at the baseline estimate */}
          <motion.line
            x1={geom.x(baseline.value)}
            x2={geom.x(baseline.value)}
            y1={ROW_B - 16}
            y2={ROW_C + 16}
            stroke="var(--muted-bar)"
            strokeWidth={1}
            strokeDasharray="2 4"
            strokeLinecap="round"
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            transition={{ delay: 0.45, duration: 0.4 }}
          />

          {/* row labels */}
          <text x={0} y={ROW_B - 8} fontSize={10} fontWeight={600} fill="var(--text-3)" letterSpacing="0.07em">
            BASELINE
          </text>
          <text x={0} y={ROW_B + 7} fontSize={12.5} fontWeight={600} fill="var(--text-2)">
            {baselineLabel}
          </text>
          <text x={0} y={ROW_C - 8} fontSize={10} fontWeight={600} fill="var(--gold-ink)" letterSpacing="0.07em">
            CANDIDATE
          </text>
          <text x={0} y={ROW_C + 7} fontSize={12.5} fontWeight={600} fill="var(--text-1)">
            {candidateLabel}
          </text>

          <Whisker est={baseline} x={geom.x} y={ROW_B} color="var(--muted-bar)" opacity={0.85} delay={0.1} />
          <Whisker est={candidate} x={geom.x} y={ROW_C} color="var(--gold-data)" opacity={1} delay={0.26} />

          {valueLabel(baseline, ROW_B, "var(--text-2)")}
          {valueLabel(candidate, ROW_C, "var(--gold-ink)")}
        </svg>
      )}
    </div>
  );
}
