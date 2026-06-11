/** Translucent chrome-glass sidebar: views, evals with ledger dots, theme. */

import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useStore } from "../data";
import { useTheme, type ThemePref } from "../theme";
import {
  IconAuto,
  IconCompare,
  IconFlask,
  IconMoon,
  IconShield,
  IconSun,
  IconTimeline,
} from "./Icons";
import { Segmented } from "./Segmented";

const THEME_OPTIONS = [
  { value: "auto" as ThemePref, icon: <IconAuto size={13} />, title: "Match system" },
  { value: "light" as ThemePref, icon: <IconSun size={13} />, title: "Light" },
  { value: "dark" as ThemePref, icon: <IconMoon size={13} />, title: "Dark" },
];

export function Sidebar() {
  const { runs, ledger, meta, evalNames } = useStore();
  const { pref, setPref } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();

  const activeEval =
    location.pathname === "/" ? (params.get("eval") ?? evalNames[0]) : null;

  const levelFor = (name: string) =>
    ledger.find((l) => l.eval_name === name)?.level ?? "ok";
  const countFor = (name: string) =>
    runs.filter((r) => r.eval_name === name).length;

  return (
    <nav className="sidebar glass-chrome">
      <div className="sidebar-brand">
        <div className="mark">
          <IconFlask size={15} />
        </div>
        <div>
          <div className="name">holdout</div>
          <div className="sub">eval run store</div>
        </div>
      </div>

      <button
        className={`side-item ${location.pathname === "/" ? "active" : ""}`}
        onClick={() => navigate("/")}
      >
        <span className="side-icon">
          <IconTimeline />
        </span>
        Timeline
      </button>
      <button
        className={`side-item ${location.pathname.startsWith("/compare") ? "active" : ""}`}
        onClick={() => navigate("/compare")}
      >
        <span className="side-icon">
          <IconCompare />
        </span>
        Compare
      </button>
      <button
        className={`side-item ${location.pathname.startsWith("/discipline") ? "active" : ""}`}
        onClick={() => navigate("/discipline")}
      >
        <span className="side-icon">
          <IconShield />
        </span>
        Discipline
      </button>

      <div className="side-section">Evals</div>
      {evalNames.map((name) => (
        <button
          key={name}
          className={`side-item ${activeEval === name ? "active" : ""}`}
          onClick={() => navigate(`/?eval=${encodeURIComponent(name)}`)}
        >
          <span className={`level-dot level-${levelFor(name)}`} />
          {name}
          <span className="count">{countFor(name)}</span>
        </button>
      ))}

      <div className="sidebar-foot">
        <Segmented options={THEME_OPTIONS} value={pref} onChange={setPref} iconsOnly />
        {meta && (
          <div className="sidebar-meta mono" title={meta.store}>
            {meta.store} · {meta.n_runs} runs · v{meta.version}
          </div>
        )}
      </div>
    </nav>
  );
}
