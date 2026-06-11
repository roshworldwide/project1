/** Timeline (home): trend chart + glass run cards for the selected eval. */

import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { CountUp } from "../components/CountUp";
import { ErrorBar } from "../components/ErrorBar";
import { GlassCard, StateCard } from "../components/Glass";
import { IconChevronRight } from "../components/Icons";
import { Segmented } from "../components/Segmented";
import { TrendChart } from "../components/TrendChart";
import { useStore } from "../data";
import { fmtCI, fmtDate, fmtPct } from "../format";
import { usePageTitle } from "../title";
import type { RunSummary } from "../api";

const list = {
  hidden: {},
  show: { transition: { staggerChildren: 0.055, delayChildren: 0.1 } },
};

const item = {
  hidden: { opacity: 0, y: 14 },
  show: {
    opacity: 1,
    y: 0,
    transition: { type: "spring", stiffness: 220, damping: 26 },
  },
} as const;

function RunCard({
  run,
  domains,
  onOpen,
}: {
  run: RunSummary;
  domains: Record<string, [number, number]>;
  onOpen: () => void;
}) {
  const metricNames = Object.keys(run.metrics);
  return (
    <motion.div variants={item}>
      <GlassCard interactive onClick={onOpen} className="run-card-wrap">
        <div className="run-card">
          <div className="who">
            <div className="target">{run.target_name}</div>
            <div className="meta-row">
              <span className="chip mono">{run.short_run_id}</span>
              <span>{fmtDate(run.created_at)}</span>
              <span className="num">n = {run.n_cases}</span>
              {run.n_errors > 0 && (
                <span className="score-chip score-fail">
                  {run.n_errors} error{run.n_errors === 1 ? "" : "s"}
                </span>
              )}
            </div>
          </div>
          <div className="metrics">
            {metricNames.map((m) => {
              const est = run.metrics[m]!;
              return (
                <div className="metric-block" key={m}>
                  <div className="label">{m.replace(/_/g, " ")}</div>
                  <div className="value">
                    <CountUp value={est.value} format={(v) => fmtPct(v)} />
                  </div>
                  <ErrorBar estimate={est} domain={domains[m]} />
                  <div className="ci-label">
                    95% CI {fmtCI(est.ci_low, est.ci_high)}
                  </div>
                </div>
              );
            })}
            <span className="card-chev">
              <IconChevronRight size={14} />
            </span>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}

export function Timeline() {
  const { runs, evalNames, loading, error } = useStore();
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const evalName = params.get("eval") ?? evalNames[0] ?? "";
  usePageTitle(evalName || "Timeline");

  const evalRuns = useMemo(
    () => runs.filter((r) => r.eval_name === evalName),
    [runs, evalName],
  );
  const ascending = useMemo(
    () =>
      [...evalRuns].sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      ),
    [evalRuns],
  );

  const metricNames = useMemo(() => {
    const names: string[] = [];
    for (const r of evalRuns)
      for (const m of Object.keys(r.metrics))
        if (!names.includes(m)) names.push(m);
    return names;
  }, [evalRuns]);

  // Shared per-metric axis domain so error bars are comparable across cards.
  const domains = useMemo(() => {
    const out: Record<string, [number, number]> = {};
    for (const m of metricNames) {
      let lo = 1;
      let hi = 0;
      for (const r of evalRuns) {
        const est = r.metrics[m];
        if (!est) continue;
        lo = Math.min(lo, est.ci_low);
        hi = Math.max(hi, est.ci_high);
      }
      out[m] = [lo, hi];
    }
    return out;
  }, [evalRuns, metricNames]);

  const [metricChoice, setMetricChoice] = useState<string | null>(null);
  const metric =
    metricChoice && metricNames.includes(metricChoice)
      ? metricChoice
      : (metricNames[0] ?? "");

  if (loading) {
    return (
      <div className="page">
        <StateCard title="Loading runs…" spinner />
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
  if (evalRuns.length === 0) {
    return (
      <div className="page">
        <h1 className="large-title">Timeline</h1>
        <StateCard
          title="No runs yet"
          body="Run an eval against this store and it will appear here."
          hint="holdout run --store .holdout"
        />
      </div>
    );
  }

  return (
    <div className="page">
      <h1 className="large-title">{evalName}</h1>
      <div className="page-sub num">
        {evalRuns.length} runs · newest first · bootstrap 95% intervals
      </div>

      <GlassCard className="chart-card" elevated>
        <div className="chart-head">
          <div>
            <div className="chart-title">{metric.replace(/_/g, " ")} over time</div>
            <div className="chart-sub">
              shaded band = 95% confidence interval per run
            </div>
          </div>
          {metricNames.length > 1 && (
            <Segmented
              options={metricNames.map((m) => ({
                value: m,
                label: m.replace(/_/g, " "),
              }))}
              value={metric}
              onChange={setMetricChoice}
            />
          )}
        </div>
        <TrendChart
          runs={ascending}
          metric={metric}
          onPick={(r) => navigate(`/runs/${r.run_id}`)}
        />
      </GlassCard>

      <motion.div
        variants={list}
        initial="hidden"
        animate="show"
        style={{ display: "flex", flexDirection: "column", gap: 14 }}
      >
        {evalRuns.map((run) => (
          <RunCard
            key={run.run_id}
            run={run}
            domains={domains}
            onOpen={() => navigate(`/runs/${run.run_id}`)}
          />
        ))}
      </motion.div>
    </div>
  );
}
