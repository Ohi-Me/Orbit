import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        paper: "#EEF0EA",
        ink: "#14181C",
        navy: "#1C2B45",
        navy2: "#28405F",
        grid: "#D7D9D0",
        gain: "#1F7A5C",
        loss: "#B23B2E",
        warn: "#C97A2B",
        muted: "#5B6259",
      },
      fontFamily: {
        display: ["var(--font-serif)", "Georgia", "serif"],
        body: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      backgroundImage: {
        "grid-paper":
          "linear-gradient(to right, rgba(28,43,69,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(28,43,69,0.06) 1px, transparent 1px)",
      },
      backgroundSize: {
        "grid-cell": "24px 24px",
      },
    },
  },
  plugins: [],
};
export default config;
