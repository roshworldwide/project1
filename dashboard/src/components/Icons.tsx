/** Minimal hand-drawn icon set: 1.5px stroke, rounded caps, outline style. */

interface IconProps {
  size?: number;
}

function Svg({
  size = 16,
  children,
}: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {children}
    </svg>
  );
}

export function IconTimeline(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M2 12.5 6 8l2.5 2.5L14 4.5" />
      <circle cx="6" cy="8" r="1" fill="currentColor" stroke="none" />
      <circle cx="8.5" cy="10.5" r="1" fill="currentColor" stroke="none" />
    </Svg>
  );
}

export function IconCompare(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M10.5 2.5 13.5 5.5 10.5 8.5" />
      <path d="M13.5 5.5H4.5" />
      <path d="M5.5 13.5 2.5 10.5 5.5 7.5" />
      <path d="M2.5 10.5h9" />
    </Svg>
  );
}

export function IconShield(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M8 1.8 13 3.6v4.1c0 3.2-2.1 5.5-5 6.5-2.9-1-5-3.3-5-6.5V3.6L8 1.8Z" />
      <path d="M5.8 7.8 7.4 9.4 10.4 6.2" />
    </Svg>
  );
}

export function IconSun(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.4M8 13.1v1.4M14.5 8h-1.4M2.9 8H1.5M12.6 3.4l-1 1M4.4 11.6l-1 1M12.6 12.6l-1-1M4.4 4.4l-1-1" />
    </Svg>
  );
}

export function IconMoon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M13.2 9.7A5.6 5.6 0 0 1 6.3 2.8 5.6 5.6 0 1 0 13.2 9.7Z" />
    </Svg>
  );
}

export function IconAuto(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="5.6" />
      <path d="M8 2.4v11.2A5.6 5.6 0 0 0 8 2.4Z" fill="currentColor" stroke="none" />
    </Svg>
  );
}

export function IconChevronDown(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m4.5 6.5 3.5 3.5 3.5-3.5" />
    </Svg>
  );
}

export function IconChevronLeft(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M9.5 3.5 5 8l4.5 4.5" />
    </Svg>
  );
}

export function IconChevronRight(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m6.5 3.5 4.5 4.5-4.5 4.5" />
    </Svg>
  );
}

export function IconArrowRight(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 8h10" />
      <path d="m9.5 4.5 3.5 3.5-3.5 3.5" />
    </Svg>
  );
}

export function IconWarning(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M8 2.4 14.3 13H1.7L8 2.4Z" />
      <path d="M8 6.5v3" />
      <circle cx="8" cy="11.3" r="0.7" fill="currentColor" stroke="none" />
    </Svg>
  );
}

export function IconFlask(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6.2 2h3.6M7 2.2v4.1L3.2 12a1.6 1.6 0 0 0 1.4 2.4h6.8a1.6 1.6 0 0 0 1.4-2.4L9 6.3V2.2" />
      <path d="M5 10.5h6" />
    </Svg>
  );
}
