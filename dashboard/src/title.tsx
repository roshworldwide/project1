/** Page title plumbing: views declare a title; the toolbar condenses it. */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

const Ctx = createContext<{
  title: string;
  setTitle: (t: string) => void;
}>({ title: "", setTitle: () => {} });

export function TitleProvider({ children }: { children: ReactNode }) {
  const [title, setTitle] = useState("");
  return <Ctx.Provider value={{ title, setTitle }}>{children}</Ctx.Provider>;
}

export function usePageTitle(title: string) {
  const { setTitle } = useContext(Ctx);
  useEffect(() => {
    setTitle(title);
    document.title = title ? `${title} — holdout` : "holdout";
  }, [title, setTitle]);
}

export function useTitle(): string {
  return useContext(Ctx).title;
}
