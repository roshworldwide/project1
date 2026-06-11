/** Thin gold progress arc: recorded uses vs budget. */

import { motion, useReducedMotion } from "framer-motion";

interface ProgressArcProps {
  fraction: number; // 0..1
  label: string; // center text
  size?: number;
}

export function ProgressArc({ fraction, label, size = 64 }: ProgressArcProps) {
  const reduced = useReducedMotion();
  const stroke = 4;
  const r = (size - stroke) / 2;
  const c = size / 2;
  const clamped = Math.max(0, Math.min(1, fraction));

  return (
    <svg width={size} height={size} aria-hidden>
      <circle
        cx={c}
        cy={c}
        r={r}
        fill="none"
        stroke="var(--grid-line)"
        strokeWidth={stroke}
      />
      <motion.circle
        cx={c}
        cy={c}
        r={r}
        fill="none"
        stroke="var(--gold)"
        strokeWidth={stroke}
        strokeLinecap="round"
        transform={`rotate(-90 ${c} ${c})`}
        style={{ pathLength: undefined }}
        initial={{ pathLength: reduced ? clamped : 0 }}
        animate={{ pathLength: clamped }}
        transition={{ type: "spring", stiffness: 60, damping: 18 }}
      />
      <text
        x={c}
        y={c + 4}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill="var(--text-1)"
        fontFamily="var(--font-mono)"
      >
        {label}
      </text>
    </svg>
  );
}
