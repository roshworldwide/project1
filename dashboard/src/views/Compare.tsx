/** Compare: baseline vs candidate with paired error bars — the signature view. */

import { motion } from "framer-motion";
import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useApi,
  type Comparison,
  type MetricComparison,
  type RunSummary,
} from "../api";
import { GlassCard, StateCard } from "../components/Glass";
import { IconArrowRight, IconWarning } from "../components/Icons";
import { PairedErrorBars } from "../components/PairedErrorBars";
import { RunPicker } from "../components/RunPicker";
import { Segmented } from "../components/Segmented";
import { VerdictPill } from "../components/VerdictPill";
import { findRun, useStore } from "../data";
import { fmtEffect, fmtP } from "../format";
import { usePageTitle } from "../title";
import type { Verdict } from "../api";

/** A faint verdict-colored bloom behind the banner's left edge. */
const BLOOM: Record<Verdict, string> = {
  regressed:
    "radial-gradient(420px circle at 8% 50%, rgba(229,72,77,0.16), transparent 70%)",
  improved:
    "radial-gradient(420px circle at 8% 50%, rgba(70,167,88,0.15), transparent 70%)",
  no_significant_change: "none",
  insufficient_data:
    "radial-gradient(420px circle at 8% 50%, rgba(201,168,118,0.16), transparent 70%)",
};

function MetricCard({
  c,
  index,
  baselineLabel,
  candidateLabel,
}: {
  c: MetricComparison;
  index: number;
  baselineLabel: string;
  candidateLabel: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 200, damping: 25, delay: index * 0.08 }}
    >
      <GlassCard className="metric-compare">
        <div className="head">
          <div className="name">{c.metric.replace(/_/g, " ")}</div>
          <VerdictPill verdict={c.verdict} small />
        </div>

        <PairedErrorBars
          baseline={c.baseline}
          candidate={c.candidate}
          baselineLabel={baselineLabel}
          candidateLabel={candidateLabel}
        />

        <div className="stat-row">
          <div className="stat">
            <div className="k">Δ effect</div>
            <div className="v mono">
              {c.result ? (
                <>
                  {fmtEffect(c.result.effect)}{" "}
                  <span className="ci">
                    [{fmtEffect(c.result.ci.ci_low)}, {fmtEffect(c.result.ci.ci_high)}]
                  </span>
                </>
              ) : (
                "—"
              )}
            </div>
          </div>
          <div className="stat">
            <div className="k">Test</div>
            <div className="v mono">{c.result?.test ?? "—"}</div>
          </div>
          <div className="stat">
            <div className="k">Corrected p</div>
            <div className="v mono">
              {c.p_adjusted !== null ? fmtP(c.p_adjusted) : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="k">Pairs</div>
            <div className="v mono">{c.n_pairs}</div>
          </div>
          {c.result?.detail && (
            <div className="stat">
              <div className="k">Detail</div>
              <div className="v" style={{ color: "var(--text-2)" }}>
                {c.result.detail}
              </div>
            </div>
          )}
          {c.note && (
            <div className="stat">
              <div className="k">Note</div>
              <div className="v" style={{ color: "var(--text-2)" }}>
                {c.note}
              </div>
            </div>
          )}
        </div>
      </GlassCard>
    </motion.div>
  );
}

export function Compare() {
  usePageTitle("Compare");
  const { runs, evalNames, loading } = useStore();
  const [params, setParams] = useSearchParams();

  const baselineParam = params.get("baseline");
  const candidateParam = params.get("candidate");

  const baselineRun = findRun(runs, baselineParam);
  const candidateRun = findRun(runs, candidateParam);

  // Eval scope: explicit param, else inferred from the chosen baseline.
  const evalName =
    baselineRun?.eval_name ?? params.get("eval") ?? evalNames[0] ?? "";
  const evalRuns = useMemo(
    () => runs.filter((r) => r.eval_name === evalName),
    [runs, evalName],
  );

  // Defaults: candidate = newest run, baseline = the one before it.
  const baseline =
    baselineRun && baselineRun.eval_name === evalName ? baselineRun : evalRuns[1];
  const candidate =
    candidateRun && candidateRun.eval_name === evalName
      ? candidateRun
      : evalRuns[0];

  const setPair = (b?: RunSummary, c?: RunSummary) => {
    const next = new URLSearchParams();
    if (b) next.set("baseline", b.run_id);
    if (c) next.set("candidate", c.run_id);
    setParams(next, { replace: true });
  };

  const ready = baseline && candidate;
  const cmp = useApi<Comparison>(
    ready
      ? `/api/compare?baseline=${baseline.run_id}&candidate=${candidate.run_id}&alpha=0.05`
      : null,
  );

  if (loading) {
    return (
      <div className="page">
        <StateCard title="Loading runs…" spinner />
      </div>
    );
  }

  if (evalRuns.length < 2) {
    return (
      <div className="page">
        <h1 className="large-title">Compare</h1>
        <StateCard
          title="Not enough runs"
          body="Comparing needs at least two runs of the same eval."
          hint="holdout run --store .holdout"
        />
      </div>
    );
  }

  const firstMetric = Object.keys(evalRuns[0]!.metrics)[0];

  return (
    <div className="page">
      <h1 className="large-title">Compare</h1>
      <div className="page-sub">
        Paired comparison on shared cases · α = 0.05 · Benjamini–Hochberg
      </div>

      <div className="compare-controls">
        {evalNames.length > 1 && (
          <Segmented
            options={evalNames.map((e) => ({ value: e, label: e }))}
            value={evalName}
            onChange={(e) => {
              const next = new URLSearchParams();
              next.set("eval", e);
              setParams(next, { replace: true });
            }}
          />
        )}
        <RunPicker
          role="Baseline"
          runs={evalRuns}
          selected={baseline}
          metric={firstMetric}
          onSelect={(r) => setPair(r, candidate)}
        />
        <span style={{ color: "var(--text-3)" }}>
          <IconArrowRight />
        </span>
        <RunPicker
          role="Candidate"
          runs={evalRuns}
          selected={candidate}
          metric={firstMetric}
          onSelect={(r) => setPair(baseline, r)}
        />
      </div>

      {cmp.loading && <StateCard title="Crunching the statistics…" spinner />}
      {cmp.error && <StateCard title="Comparison failed" body={cmp.error} />}

      {cmp.data && (
        <>
          <GlassCard className="verdict-banner" elevated>
            <span
              className="verdict-bloom"
              style={{ background: BLOOM[cmp.data.verdict] }}
              aria-hidden
            />
            <VerdictPill verdict={cmp.data.verdict} />
            <div className="meta">
              <strong style={{ color: "var(--text-1)" }}>
                {cmp.data.baseline_target}
              </strong>{" "}
              →{" "}
              <strong style={{ color: "var(--gold-ink)" }}>
                {cmp.data.candidate_target}
              </strong>{" "}
              on <span className="mono">{cmp.data.eval_name}</span>
              <br />
              α = {cmp.data.alpha} · {cmp.data.correction} ·{" "}
              {cmp.data.comparisons.length} metrics
            </div>
          </GlassCard>

          {cmp.data.comparisons.map((c, i) => (
            <MetricCard
              key={c.metric}
              c={c}
              index={i}
              baselineLabel={cmp.data!.baseline_target}
              candidateLabel={cmp.data!.candidate_target}
            />
          ))}

          {cmp.data.warnings.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 4 }}>
              {cmp.data.warnings.map((w) => (
                <div className="warning-row" key={w}>
                  <IconWarning size={14} />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
