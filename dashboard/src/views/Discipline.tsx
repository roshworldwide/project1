/** Discipline: holdout-ledger budgets per eval, with the Dwork framing. */

import { motion } from "framer-motion";
import { GlassCard, StateCard } from "../components/Glass";
import { ProgressArc } from "../components/ProgressArc";
import { useStore } from "../data";
import { usePageTitle } from "../title";

const LEVEL_LABEL: Record<string, string> = {
  ok: "OK",
  caution: "Caution",
  "overfit-risk": "Overfit Risk",
};

export function Discipline() {
  usePageTitle("Discipline");
  const { ledger, loading, error } = useStore();

  if (loading) {
    return (
      <div className="page">
        <StateCard title="Loading ledger…" spinner />
      </div>
    );
  }
  if (error) {
    return (
      <div className="page">
        <StateCard title="Couldn't reach the store" body={error} />
      </div>
    );
  }

  return (
    <div className="page">
      <h1 className="large-title">Discipline</h1>
      <div className="page-sub">
        Recorded adaptive uses of each holdout, against its budget.
      </div>

      {ledger.length === 0 ? (
        <StateCard
          title="No ledger entries"
          body="Recorded uses will appear once evals are run against this store."
        />
      ) : (
        <div className="ledger-grid">
          {ledger.map((entry, i) => (
            <motion.div
              key={entry.eval_fingerprint}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 220, damping: 25, delay: i * 0.07 }}
            >
              <GlassCard className="ledger-card" interactive>
                <ProgressArc
                  fraction={entry.uses / Math.max(1, entry.budget)}
                  label={`${entry.uses}`}
                />
                <div style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 5 }}>
                  <div className="name">{entry.eval_name}</div>
                  <div className="uses num">
                    {entry.uses} of {entry.budget} uses
                  </div>
                  <div>
                    <span className={`level-badge ${entry.level}`}>
                      <span className={`level-dot level-${entry.level}`} />
                      {LEVEL_LABEL[entry.level] ?? entry.level}
                    </span>
                  </div>
                  <div className="fp" title={entry.eval_fingerprint}>
                    {entry.eval_fingerprint.slice(0, 16)}
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>
      )}

      <p className="discipline-caption">
        Each adaptive look at the same eval is overfitting risk — budget your
        reads of the holdout (Dwork et&nbsp;al., <em>The Reusable Holdout</em>).
      </p>
    </div>
  );
}
