import type { Metadata } from "next";
import "./globals.css";
import Shell from "./components/Shell";

export const metadata: Metadata = {
  title: "Orbit",
  description:
    "End-to-end applied machine learning demonstrated on financial data: multi-agent pipeline orchestration with a validation feedback loop, point-in-time feature engineering, purged walk-forward model comparison across linear, gradient-boosting and deep sequence models, hybrid retrieval-augmented generation with citation and numeric verification, drift monitoring and experiment tracking.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-body text-ink antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
