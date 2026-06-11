/** Typed client for the read-only holdout dashboard API. */

import { useEffect, useState } from "react";

export interface Estimate {
  value: number;
  ci_low: number;
  ci_high: number;
  n: number;
  level: number;
  method: string;
}

export interface RunSummary {
  run_id: string;
  short_run_id: string;
  eval_name: string;
  target_name: string;
  created_at: string;
  n_cases: number;
  n_errors: number;
  seed: number | null;
  metrics: Record<string, Estimate>;
}

export interface Score {
  value: number;
  kind: string;
  detail: string | null;
}

export interface CaseResult {
  case_id: string;
  output: string | null;
  scores: Record<string, Score>;
  error: string | null;
  latency_s: number | null;
}

export interface RunDetail {
  run_id: string;
  short_run_id: string;
  eval_name: string;
  eval_fingerprint: string;
  target_name: string;
  scorer_names: string[];
  seed: number | null;
  created_at: string;
  holdout_version: string;
  n_errors: number;
  results: CaseResult[];
  metrics: Record<string, Estimate>;
}

export type Verdict =
  | "improved"
  | "regressed"
  | "no_significant_change"
  | "insufficient_data";

export interface TestResult {
  test: string;
  p_value: number;
  effect: number;
  ci: Estimate;
  n: number;
  detail: string | null;
}

export interface MetricComparison {
  metric: string;
  verdict: Verdict;
  n_pairs: number;
  result: TestResult | null;
  p_adjusted: number | null;
  baseline: Estimate;
  candidate: Estimate;
  note: string | null;
}

export interface Comparison {
  eval_name: string;
  baseline_run_id: string;
  candidate_run_id: string;
  baseline_target: string;
  candidate_target: string;
  alpha: number;
  correction: string;
  verdict: Verdict;
  comparisons: MetricComparison[];
  warnings: string[];
}

export interface LedgerEntry {
  eval_name: string;
  eval_fingerprint: string;
  uses: number;
  budget: number;
  level: "ok" | "caution" | "overfit-risk";
}

export interface Meta {
  version: string;
  store: string;
  n_runs: number;
}

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Fetch a JSON API path; `null` path means "don't fetch yet". */
export function useApi<T>(path: string | null): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    error: null,
    loading: path !== null,
  });

  useEffect(() => {
    if (path === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    fetch(path, { signal: ctrl.signal })
      .then(async (res) => {
        const body: unknown = await res.json();
        if (!res.ok) {
          const msg =
            typeof body === "object" && body !== null && "error" in body
              ? String((body as { error: unknown }).error)
              : `HTTP ${res.status}`;
          throw new Error(msg);
        }
        setState({ data: body as T, error: null, loading: false });
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setState({
          data: null,
          error: err instanceof Error ? err.message : String(err),
          loading: false,
        });
      });
    return () => ctrl.abort();
  }, [path]);

  return state;
}
