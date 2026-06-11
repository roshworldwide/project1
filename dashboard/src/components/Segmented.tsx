/** Apple-style segmented control with a sprung sliding thumb. */

import { motion } from "framer-motion";
import { useId, type ReactNode } from "react";

export interface SegmentOption<T extends string> {
  value: T;
  label?: string;
  icon?: ReactNode;
  title?: string;
}

interface SegmentedProps<T extends string> {
  options: readonly SegmentOption<T>[];
  value: T;
  onChange: (v: T) => void;
  iconsOnly?: boolean;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  iconsOnly = false,
}: SegmentedProps<T>) {
  const group = useId();
  return (
    <div
      className={`segmented ${iconsOnly ? "icons" : ""}`}
      role="radiogroup"
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            className={`segment ${active ? "active" : ""}`}
            role="radio"
            aria-checked={active}
            title={opt.title ?? opt.label}
            onClick={() => onChange(opt.value)}
          >
            {active && (
              <motion.span
                layoutId={`thumb-${group}`}
                className="segment-thumb"
                transition={{ type: "spring", stiffness: 480, damping: 36 }}
              />
            )}
            <span style={{ position: "relative", zIndex: 1, display: "inline-flex", alignItems: "center", gap: 6 }}>
              {opt.icon}
              {opt.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
