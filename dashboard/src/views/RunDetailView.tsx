/** Run detail: metric estimates with error bars + the per-case results table. */

import { motion } from "framer-motion";
import { useNavigate, useParams } from "react-router-dom";
import { useApi, type RunDetail, type Score } from "../api";
import { CountUp } from "../components/CountUp";
import { ErrorBar } from "../components/ErrorBar";
import { GlassCard, StateCard } from "../components/Glass";
import { IconChevronLeft } from "../components/Icons";
import { fmtCI, fmtDate, fmtLatency, fmtPct } from "../format";
import { usePageTitle } from "../title";

function ScoreChip({ score }: { score: Score | undefined }) {
  if (!score) return <span className="score-chip score-num">—</span>;
  if (score.kind === "binary") {
    return score.value >= 1 ? (
      <span className="score-chip score-pass" title={score.detail ?? undefined}>
        ✓
      </span>
    ) : (
      <span className="score-chip score-fail" title={score.detail ?? undefined}>
        ✕
      </span>
    );
  }
  return (
    <span className="score-chip score-num" title={score.detail ?? undefined}>
      {score.value.toFixed(3)}
    </span>
  );
}

export function RunDetailView() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const { data, error, loading } = useApi<RunDetail>(
    runId ? `/api/runs/${runId}` : null,
  );

  usePageTitle(data ? data.target_name : "Run");

  if (loading) {
    return (
      <div className="page">
        <StateCard title="Loading run…" spinner />
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="page">
        <StateCard title="Run not found" body={error ?? undefined} />
      </div>
    );
  }

  const metricNames = Object.keys(data.metrics);

  return (
    <div className="page">
      <button
        className="chip"
        style={{ marginTop: 12, gap: 4 }}
        onClick={() => navigate(-1)}
      >
        <IconChevronLeft size={11} />
        Back
      </button>
      <h1 className="large-title">{data.target_name}</h1>
      <div className="page-sub">
        <span className="mono">{data.short_run_id}</span> ·{" "}
        <span className="mono">{data.eval_name}</span> ·{" "}
        {fmtDate(data.created_at)} · n = {data.results.length}
        {data.seed !== null && <> · seed {data.seed}</>}
        {data.n_errors > 0 && (
          <>
            {" "}
            ·{" "}
            <span style={{ color: "var(--red)" }}>
              {data.n_errors} error{data.n_errors === 1 ? "" : "s"}
            </span>
          </>
        )}
      </div>

      <div className="detail-grid">
        {metricNames.map((m, i) => {
          const est = data.metrics[m]!;
          return (
            <motion.div
              key={m}
              style={{ flex: "1 1 240px", maxWidth: 340 }}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 220, damping: 25, delay: i * 0.07 }}
            >
              <GlassCard className="card-pad" elevated>
                <div className="metric-block" style={{ width: "100%" }}>
                  <div className="label">{m.replace(/_/g, " ")}</div>
                  <div className="value" style={{ fontSize: 28 }}>
                    <CountUp value={est.value} format={(v) => fmtPct(v)} />
                  </div>
                  <ErrorBar estimate={est} height={16} />
                  <div className="ci-label">
                    95% CI {fmtCI(est.ci_low, est.ci_high)} · {est.method}
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          );
        })}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 180, damping: 24, delay: 0.15 }}
      >
        <GlassCard>
          <table className="results-table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Output</th>
                {data.scorer_names.map((s) => (
                  <th key={s}>{s.replace(/_/g, " ")}</th>
                ))}
                <th style={{ textAlign: "right" }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r) => (
                <tr key={r.case_id} className={r.error ? "error-row" : ""}>
                  <td className="mono" style={{ color: "var(--text-2)" }}>
                    {r.case_id}
                  </td>
                  <td className="output-cell" title={r.error ?? r.output ?? ""}>
                    {r.error ?? r.output ?? "—"}
                  </td>
                  {data.scorer_names.map((s) => (
                    <td key={s}>
                      <ScoreChip score={r.scores[s]} />
                    </td>
                  ))}
                  <td
                    className="mono"
                    style={{ textAlign: "right", color: "var(--text-3)" }}
                  >
                    {fmtLatency(r.latency_s)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>
      </motion.div>
    </div>
  );
}
