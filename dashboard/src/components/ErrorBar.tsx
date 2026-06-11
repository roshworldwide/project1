/** A compact inline CI error bar: track, whisker with end caps, value dot.
 *  Draws in with a spring, scaling outward from the value point. */

import { motion } from "framer-motion";
import type { Estimate } from "../api";
import { useMeasure } from "../hooks";

interface ErrorBarProps {
  estimate: Estimate;
  /** Axis domain; defaults to a padded window around the CI itself. */
  domain?: [number, number];
  color?: string;
  height?: number;
}

export function ErrorBar({
  estimate,
  domain,
  color = "var(--gold-data)",
  height = 14,
}: ErrorBarProps) {
  const [ref, width] = useMeasure<HTMLDivElement>();
  const { value, ci_low, ci_high } = estimate;
  let [lo, hi] = domain ?? [ci_low, ci_high];
  if (hi - lo < 0.02) {
    lo -= 0.03;
    hi += 0.03;
  }
  const padD = Math.max(0.01, (hi - lo) * 0.14);
  lo = Math.max(0, lo - padD);
  hi = Math.min(1, hi + padD);
  const pad = 5;
  const x = (v: number) =>
    pad + ((v - lo) / (hi - lo)) * (width - pad * 2);
  const cy = height / 2;
  const cap = 4;

  return (
    <div ref={ref} style={{ width: "100%" }}>
      {width > 0 && (
        <svg width={width} height={height} aria-hidden>
          {/* full-domain track */}
          <line
            x1={pad}
            x2={width - pad}
            y1={cy}
            y2={cy}
            stroke="var(--grid-line)"
            strokeWidth={2}
            strokeLinecap="round"
          />
          <g transform={`translate(${x(value)}, ${cy})`}>
            <motion.g
              initial={{ scaleX: 0, opacity: 0 }}
              animate={{ scaleX: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 120, damping: 20 }}
            >
              <line
                x1={x(ci_low) - x(value)}
                x2={x(ci_high) - x(value)}
                y1={0}
                y2={0}
                stroke={color}
                strokeOpacity={0.6}
                strokeWidth={1.5}
                strokeLinecap="round"
              />
              <line
                x1={x(ci_low) - x(value)}
                x2={x(ci_low) - x(value)}
                y1={-cap}
                y2={cap}
                stroke={color}
                strokeOpacity={0.6}
                strokeWidth={1.5}
                strokeLinecap="round"
              />
              <line
                x1={x(ci_high) - x(value)}
                x2={x(ci_high) - x(value)}
                y1={-cap}
                y2={cap}
                stroke={color}
                strokeOpacity={0.6}
                strokeWidth={1.5}
                strokeLinecap="round"
              />
            </motion.g>
            <motion.circle
              r={3.2}
              fill={color}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 18 }}
            />
          </g>
        </svg>
      )}
    </div>
  );
}
