/** A number that springs from 0 to its value on first render. */

import {
  useMotionValueEvent,
  useReducedMotion,
  useSpring,
} from "framer-motion";
import { useEffect, useState } from "react";

interface CountUpProps {
  value: number;
  format: (v: number) => string;
}

export function CountUp({ value, format }: CountUpProps) {
  const reduced = useReducedMotion();
  const spring = useSpring(reduced ? value : 0, {
    stiffness: 90,
    damping: 24,
    mass: 0.8,
  });
  const [shown, setShown] = useState(reduced ? value : 0);

  useEffect(() => {
    if (reduced) spring.jump(value);
    else spring.set(value);
  }, [value, reduced, spring]);

  useMotionValueEvent(spring, "change", (v) => setShown(v));

  return <span className="num">{format(shown)}</span>;
}
