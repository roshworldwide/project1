/** Shared store data: runs listing, ledger, and meta, fetched once. */

import { createContext, useContext, useMemo, type ReactNode } from "react";
import {
  useApi,
  type LedgerEntry,
  type Meta,
  type RunSummary,
} from "./api";

export interface StoreData {
  runs: RunSummary[];
  ledger: LedgerEntry[];
  meta: Meta | null;
  evalNames: string[];
  loading: boolean;
  error: string | null;
}

const Ctx = createContext<StoreData>({
  runs: [],
  ledger: [],
  meta: null,
  evalNames: [],
  loading: true,
  error: null,
});

export function DataProvider({ children }: { children: ReactNode }) {
  const runsState = useApi<{ runs: RunSummary[] }>("/api/runs");
  const ledgerState = useApi<{ evals: LedgerEntry[] }>("/api/ledger");
  const metaState = useApi<Meta>("/api/meta");

  const value = useMemo<StoreData>(() => {
    const runs = runsState.data?.runs ?? [];
    const evalNames: string[] = [];
    for (const r of runs) {
      if (!evalNames.includes(r.eval_name)) evalNames.push(r.eval_name);
    }
    return {
      runs,
      ledger: ledgerState.data?.evals ?? [],
      meta: metaState.data,
      evalNames,
      loading: runsState.loading || ledgerState.loading,
      error: runsState.error ?? ledgerState.error,
    };
  }, [runsState, ledgerState, metaState]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): StoreData {
  return useContext(Ctx);
}

/** Resolve a run by id or id prefix. */
export function findRun(
  runs: RunSummary[],
  ref: string | null,
): RunSummary | undefined {
  if (!ref) return undefined;
  return runs.find((r) => r.run_id === ref || r.run_id.startsWith(ref));
}
