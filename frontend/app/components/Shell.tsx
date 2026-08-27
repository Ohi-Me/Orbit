"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { clearSession, getApprovals, getHealth, getUser } from "../lib/api";
import type { Health } from "../lib/types";
import { Badge } from "./ui";

const NAV = [
  { href: "/", label: "Pipelines" },
  { href: "/runs", label: "Experiments" },
  { href: "/models", label: "Models" },
  { href: "/documents", label: "Documents" },
  { href: "/agents", label: "Agents" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/monitoring", label: "Monitoring" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [pending, setPending] = useState(0);
  const [user, setUser] = useState<ReturnType<typeof getUser>>(null);

  useEffect(() => {
    const sync = () => setUser(getUser());
    sync();
    window.addEventListener("quant-auth-change", sync);
    return () => window.removeEventListener("quant-auth-change", sync);
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (alive) {
          setHealth(h);
          setOffline(false);
        }
      } catch {
        if (alive) setOffline(true);
      }
      try {
        const a = await getApprovals("pending");
        if (alive) setPending(a.n_pending);
      } catch {
        /* approvals are non-critical for the shell */
      }
    };
    poll();
    const id = setInterval(poll, 20000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pathname]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-grid bg-paper/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-5 gap-y-2 px-5 py-2.5">
          <Link href="/" className="leading-none">
            <span className="font-display text-base text-navy">Orbit</span>
            <span className="ml-2 hidden text-[10px] uppercase tracking-wider text-muted sm:inline">
              applied ML · financial data
            </span>
          </Link>

          <nav className="flex flex-wrap items-center gap-1">
            {NAV.map((item) => {
              const active =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-2 py-1 text-xs uppercase tracking-wider transition-colors ${
                    active ? "text-navy underline underline-offset-4" : "text-muted hover:text-navy"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {pending > 0 && (
              <Link
                href="/approvals"
                className="flex items-center gap-1 border border-warn/50 bg-warn/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-warn hover:bg-warn/20"
                title={`${pending} run(s) awaiting a human decision`}
              >
                {pending} pending review
              </Link>
            )}

            {offline ? (
              <Badge tone="fail" title="The API is not reachable">
                api offline
              </Badge>
            ) : health ? (
              <Badge
                tone={health.fidelity === "full" ? "ok" : "warn"}
                title={
                  health.fidelity === "full"
                    ? "All capabilities probed available: live market data, filings, macro, neural embeddings, domain transformer, deep learning."
                    : `Degraded: ${health.degraded_capabilities.join(", ")}`
                }
              >
                {health.fidelity === "full" ? "full fidelity" : "degraded"}
              </Badge>
            ) : null}

            {user ? (
              <div className="flex items-center gap-2">
                <span className="hidden text-xs text-muted md:inline">{user.email}</span>
                <button
                  onClick={() => clearSession()}
                  className="text-xs uppercase tracking-wider text-muted hover:text-loss"
                >
                  sign out
                </button>
              </div>
            ) : (
              <Link
                href="/login"
                className="text-xs uppercase tracking-wider text-navy hover:underline"
              >
                sign in
              </Link>
            )}
          </div>
        </div>

        {health?.warnings?.length ? (
          <div className="border-t border-warn/30 bg-warn/10 px-5 py-1 text-[11px] text-warn">
            {health.warnings.join(" · ")}
          </div>
        ) : null}
      </header>

      <main className="mx-auto max-w-[1400px] px-5 py-6">{children}</main>

      <footer className="mx-auto max-w-[1400px] px-5 pb-10 pt-4 text-[11px] leading-relaxed text-muted">
        An applied machine-learning platform demonstrated on financial data. Outputs are research
        artefacts, not advice. Every model result is reported with its validation scheme, its
        corrected significance, and its stated limitations — and nothing is treated as
        decision-grade until a named person approves it.
      </footer>
    </div>
  );
}
