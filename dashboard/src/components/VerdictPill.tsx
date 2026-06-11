/** Verdict pill: REGRESSED / IMPROVED / NO SIGNIFICANT CHANGE / INSUFFICIENT DATA. */

import { motion } from "framer-motion";
import type { Verdict } from "../api";

const STYLES: Record<
  Verdict,
  { label: string; color: string; bg: string; glow: string }
> = {
  regressed: {
    label: "Regressed",
    color: "var(--red)",
    bg: "rgba(229, 72, 77, 0.13)",
    glow: "0 0 28px rgba(229, 72, 77, 0.28)",
  },
  improved: {
    label: "Improved",
    color: "var(--green)",
    bg: "rgba(70, 167, 88, 0.13)",
    glow: "0 0 28px rgba(70, 167, 88, 0.25)",
  },
  no_significant_change: {
    label: "No Significant Change",
    color: "var(--text-2)",
    bg: "var(--chip-bg)",
    glow: "none",
  },
  insufficient_data: {
    label: "Insufficient Data",
    color: "var(--gold-ink)",
    bg: "rgba(201, 168, 118, 0.15)",
    glow: "0 0 28px rgba(201, 168, 118, 0.25)",
  },
};

export function VerdictPill({
  verdict,
  small = false,
}: {
  verdict: Verdict;
  small?: boolean;
}) {
  const s = STYLES[verdict];
  return (
    <motion.span
      className={`verdict-pill ${small ? "small" : ""}`}
      style={{
        color: s.color,
        background: s.bg,
        boxShadow: small ? "none" : `inset 0 0 0 1px ${s.color}22, ${s.glow}`,
      }}
      initial={{ scale: 0.85, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 320, damping: 20 }}
    >
      <span className="dot" />
      {s.label}
    </motion.span>
  );
}
