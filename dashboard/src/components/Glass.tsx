/** Glass primitives: cards with a cursor-following specular highlight. */

import { motion } from "framer-motion";
import { useRef, type ReactNode } from "react";

const lift = { type: "spring", stiffness: 350, damping: 26 } as const;

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  elevated?: boolean;
  interactive?: boolean;
  onClick?: () => void;
  layout?: boolean;
}

export function GlassCard({
  children,
  className = "",
  elevated = false,
  interactive = false,
  onClick,
}: GlassCardProps) {
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (e: React.PointerEvent) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - r.left}px`);
    el.style.setProperty("--my", `${e.clientY - r.top}px`);
  };

  return (
    <motion.div
      ref={ref}
      className={`card ${elevated ? "glass-elev" : "glass"} ${className}`}
      onPointerMove={interactive ? onMove : undefined}
      onClick={onClick}
      whileHover={interactive ? { y: -2, scale: 1.004 } : undefined}
      whileTap={interactive && onClick ? { scale: 0.992 } : undefined}
      transition={lift}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
    >
      {children}
    </motion.div>
  );
}

export function StateCard({
  title,
  body,
  hint,
  spinner = false,
}: {
  title: string;
  body?: string;
  hint?: string;
  spinner?: boolean;
}) {
  return (
    <motion.div
      className="card glass state-card"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 160, damping: 22 }}
    >
      {spinner && <div className="spinner" />}
      <div className="title">{title}</div>
      {body && <div>{body}</div>}
      {hint && <div className="hint">{hint}</div>}
    </motion.div>
  );
}
