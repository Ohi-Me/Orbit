"use client";

import React from "react";

/* ------------------------------------------------------------------ format */

export const pct = (v: number | null | undefined, nd = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(nd)}%`;

export const num = (v: number | null | undefined, nd = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(nd);

export const int = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString();

export const shortDate = (iso?: string | null) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "—";

export const relTime = (iso?: string | null) => {
  if (!iso) return "—";
  const secs = (Date.now() - new Date(iso).getTime()) / 1000;
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
};

/* -------------------------------------------------------------------- Card */

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`border border-grid bg-white/70 ${className}`}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-grid px-4 py-3">
          <div>
            {title && <h2 className="font-display text-lg leading-tight text-navy">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-muted">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------- Stat */

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  tone?: "neutral" | "gain" | "loss" | "warn";
}) {
  const toneClass =
    tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : tone === "warn" ? "text-warn" : "text-ink";
  return (
    <div className="border border-grid bg-paper/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`font-mono text-lg ${toneClass}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[10px] leading-snug text-muted">{hint}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ Badge */

const BADGE_TONES: Record<string, string> = {
  ok: "border-gain/40 bg-gain/10 text-gain",
  gain: "border-gain/40 bg-gain/10 text-gain",
  fail: "border-loss/40 bg-loss/10 text-loss",
  loss: "border-loss/40 bg-loss/10 text-loss",
  warn: "border-warn/40 bg-warn/10 text-warn",
  info: "border-navy/30 bg-navy/5 text-navy",
  muted: "border-grid bg-paper text-muted",
};

export function Badge({
  children,
  tone = "muted",
  title,
}: {
  children: React.ReactNode;
  tone?: keyof typeof BADGE_TONES;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-block whitespace-nowrap border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${BADGE_TONES[tone] ?? BADGE_TONES.muted}`}
    >
      {children}
    </span>
  );
}

export function statusTone(status?: string | null): keyof typeof BADGE_TONES {
  switch (status) {
    case "approved":
    case "ok":
      return "ok";
    case "failed":
    case "rejected":
      return "fail";
    case "running":
    case "queued":
    case "degraded":
      return "warn";
    case "awaiting_approval":
      return "info";
    default:
      return "muted";
  }
}

export function verdictTone(verdict?: string | null): keyof typeof BADGE_TONES {
  if (!verdict) return "muted";
  if (verdict.startsWith("REJECT")) return "fail";
  if (verdict.startsWith("CAUTION")) return "warn";
  if (verdict.startsWith("NO_STRUCTURAL")) return "ok";
  return "muted";
}

/* ------------------------------------------------------------------ Table */

export function Table({
  head,
  children,
  dense = false,
}: {
  head: React.ReactNode[];
  children: React.ReactNode;
  dense?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-grid text-left">
            {head.map((h, i) => (
              <th
                key={i}
                className={`whitespace-nowrap px-2 ${dense ? "py-1" : "py-2"} text-[10px] font-semibold uppercase tracking-wider text-muted`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Td({
  children,
  mono = false,
  className = "",
}: {
  children: React.ReactNode;
  mono?: boolean;
  className?: string;
}) {
  return (
    <td className={`px-2 py-1.5 align-top ${mono ? "font-mono text-xs" : ""} ${className}`}>
      {children}
    </td>
  );
}

/* ------------------------------------------------------------- primitives */

export function Button({
  children,
  variant = "primary",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" }) {
  const base =
    "inline-flex items-center justify-center border px-3 py-1.5 text-xs uppercase tracking-wider transition-colors disabled:cursor-not-allowed disabled:opacity-40";
  const variants = {
    primary: "border-navy bg-navy text-paper hover:bg-navy2",
    ghost: "border-grid bg-transparent text-navy hover:bg-navy/5",
    danger: "border-loss/50 bg-transparent text-loss hover:bg-loss/10",
  };
  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted">
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-navy/30 border-t-navy" />
      {label}
    </div>
  );
}

export function ErrorNote({ children }: { children: React.ReactNode }) {
  return (
    <div className="border border-loss/40 bg-loss/5 px-3 py-2 text-xs text-loss">{children}</div>
  );
}

export function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="border-l-2 border-navy/25 pl-3 text-xs leading-relaxed text-muted">{children}</p>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="px-2 py-6 text-center text-xs text-muted">{children}</div>;
}

/* ------------------------------------------------------------------- Tabs */

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: string; label: string; badge?: React.ReactNode }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-0 border-b border-grid">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`-mb-px flex items-center gap-1.5 border-b-2 px-3 py-2 text-xs uppercase tracking-wider transition-colors ${
            active === t.id
              ? "border-navy text-navy"
              : "border-transparent text-muted hover:text-navy"
          }`}
        >
          {t.label}
          {t.badge}
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------------- data-integrity */

export function SyntheticWarning({ isSynthetic }: { isSynthetic?: boolean | null }) {
  if (!isSynthetic) return null;
  return (
    <div className="border border-warn/50 bg-warn/10 px-3 py-2 text-xs text-warn">
      <strong className="uppercase tracking-wide">Synthetic data.</strong> This run used a
      generated price series, not a market. The methodology is demonstrable; the performance
      figures carry no information about real assets.
    </div>
  );
}

/** Renders a value alongside the naive alternative, so the gap is visible. */
export function CorrectedValue({
  corrected,
  naive,
  label,
}: {
  corrected: number | null | undefined;
  naive: number | null | undefined;
  label?: string;
}) {
  return (
    <span className="font-mono text-xs">
      <span className="text-ink">{num(corrected)}</span>
      {naive !== null && naive !== undefined && (
        <span className="ml-1 text-muted line-through" title={label ?? "uncorrected value"}>
          {num(naive)}
        </span>
      )}
    </span>
  );
}
