/** Unified translucent toolbar; the page's large title condenses into it. */

import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "../data";
import { useTitle } from "../title";

export function Toolbar({ condensed }: { condensed: boolean }) {
  const title = useTitle();
  const { meta } = useStore();

  return (
    <header className={`toolbar ${condensed ? "condensed" : ""}`}>
      <AnimatePresence>
        {condensed && (
          <motion.div
            key="t"
            className="toolbar-title"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ type: "spring", stiffness: 420, damping: 32 }}
          >
            {title}
          </motion.div>
        )}
      </AnimatePresence>
      <div className="toolbar-right">
        {meta && (
          <span className="mono">
            read-only · local · {meta.n_runs} runs
          </span>
        )}
      </div>
    </header>
  );
}
