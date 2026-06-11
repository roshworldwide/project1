/** Glass popover picker for choosing a run (baseline or candidate). */

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { RunSummary } from "../api";
import { fmtDate, fmtPct } from "../format";
import { IconChevronDown } from "./Icons";

interface RunPickerProps {
  role: string;
  runs: RunSummary[]; // candidates to choose from (same eval), newest first
  selected: RunSummary | undefined;
  metric?: string;
  onSelect: (run: RunSummary) => void;
}

export function RunPicker({ role, runs, selected, metric, onSelect }: RunPickerProps) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={wrap} style={{ position: "relative" }}>
      <button
        className="picker-btn glass card"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <div>
          <div className="role">{role}</div>
          <div className="who">{selected?.target_name ?? "Choose a run"}</div>
          {selected && (
            <div className="sub mono">
              {selected.short_run_id} · {fmtDate(selected.created_at)}
            </div>
          )}
        </div>
        <motion.span
          className="chev"
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ type: "spring", stiffness: 400, damping: 28 }}
        >
          <IconChevronDown size={13} />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="popover glass-elev"
            initial={{ opacity: 0, scale: 0.94, y: -6 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -4 }}
            transition={{ type: "spring", stiffness: 460, damping: 32 }}
            style={{ transformOrigin: "top left" }}
          >
            {runs.map((r) => (
              <button
                key={r.run_id}
                className={`popover-item ${r.run_id === selected?.run_id ? "selected" : ""}`}
                onClick={() => {
                  onSelect(r);
                  setOpen(false);
                }}
              >
                <div>
                  <div className="who">{r.target_name}</div>
                  <div className="sub mono">
                    {r.short_run_id} · {fmtDate(r.created_at)}
                  </div>
                </div>
                {metric && r.metrics[metric] && (
                  <span className="val">{fmtPct(r.metrics[metric]!.value)}</span>
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
