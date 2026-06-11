/** App shell: ambient background, chrome sidebar, toolbar, routed views. */

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useRef, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { Toolbar } from "./components/Toolbar";
import { DataProvider } from "./data";
import { ThemeProvider } from "./theme";
import { TitleProvider } from "./title";
import { Compare } from "./views/Compare";
import { Discipline } from "./views/Discipline";
import { RunDetailView } from "./views/RunDetailView";
import { Timeline } from "./views/Timeline";

function Ambient() {
  return (
    <div className="ambient" aria-hidden>
      <div className="blob blob-gold" />
      <div className="blob blob-blue" />
      <div className="blob blob-violet" />
      <div className="blob blob-teal" />
    </div>
  );
}

function Shell() {
  const location = useLocation();
  const [condensed, setCondensed] = useState(false);
  const raf = useRef(0);

  const onScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const top = e.currentTarget.scrollTop;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => setCondensed(top > 44));
  }, []);

  return (
    <div className="app">
      <Sidebar />
      <div className="main" onScroll={onScroll}>
        <Toolbar condensed={condensed} />
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname + (location.search || "")}
            initial={{ opacity: 0, scale: 0.992, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.996, y: -6 }}
            transition={{ type: "spring", stiffness: 260, damping: 30 }}
          >
            <Routes location={location}>
              <Route path="/" element={<Timeline />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/runs/:runId" element={<RunDetailView />} />
              <Route path="/discipline" element={<Discipline />} />
              <Route path="*" element={<Timeline />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <DataProvider>
        <TitleProvider>
          <Ambient />
          <Shell />
        </TitleProvider>
      </DataProvider>
    </ThemeProvider>
  );
}
