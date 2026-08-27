"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { ApiError, login, setSession, signup } from "../lib/api";
import { Button, Card, ErrorNote, Note } from "../components/ui";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res =
        mode === "login" ? await login(email, password) : await signup(email, password, displayName);
      setSession(res.access_token, res.user);
      router.push("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-md">
      <Card
        title={mode === "login" ? "Sign in" : "Create an account"}
        subtitle="Approvals must be attributable to a named person, so recording a decision requires an account."
      >
        <form onSubmit={submit} className="space-y-3">
          {mode === "signup" && (
            <label className="block">
              <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">Name</span>
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full border border-grid bg-white px-3 py-2 text-sm outline-none focus:border-navy"
              />
            </label>
          )}
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full border border-grid bg-white px-3 py-2 text-sm outline-none focus:border-navy"
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wider text-muted">
              Password {mode === "signup" && <span className="normal-case">(min 8 characters)</span>}
            </span>
            <input
              type="password"
              required
              minLength={mode === "signup" ? 8 : 1}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-grid bg-white px-3 py-2 text-sm outline-none focus:border-navy"
            />
          </label>

          {error && <ErrorNote>{error}</ErrorNote>}

          <Button type="submit" disabled={busy} className="w-full">
            {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
          </Button>
        </form>

        <button
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError(null);
          }}
          className="mt-3 w-full text-center text-xs text-muted hover:text-navy"
        >
          {mode === "login" ? "Need an account? Sign up" : "Already have an account? Sign in"}
        </button>

        <div className="mt-4">
          <Note>
            Browsing research is open when the server runs with AUTH_REQUIRED off. Signing in is
            what lets you approve or reject a run and adopt weights into a book.
          </Note>
        </div>
      </Card>
    </div>
  );
}
