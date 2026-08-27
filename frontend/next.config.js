/** @type {import('next').NextConfig} */

// Optional same-origin API proxy.
//
// Setting API_PROXY_TARGET makes the Next server forward /api/* to the
// backend, so the browser only ever talks to the origin it loaded from. What
// that buys you:
//   * no cross-origin request at all, so CORS_ORIGINS never needs configuring
//   * the backend URL never appears in the client bundle
//
// IMPORTANT, and verified rather than assumed: this is NOT a way to avoid
// rebuilding. Next serializes rewrites into .next/routes-manifest.json during
// `next build`, and `next start` reads that manifest -- it does not re-evaluate
// this file at boot. So API_PROXY_TARGET must be present at BUILD time, exactly
// like NEXT_PUBLIC_API_BASE. Setting either one on a already-built deployment
// changes nothing until it is rebuilt.
//
// Use NEXT_PUBLIC_API_BASE when the API is on a public URL and CORS is already
// configured; use API_PROXY_TARGET when you would rather not deal with CORS.
// NEXT_PUBLIC_API_BASE takes precedence in the client when both are set.
const proxyTarget = (process.env.API_PROXY_TARGET || "").replace(/\/$/, "");

const nextConfig = {
  reactStrictMode: true,

  async rewrites() {
    if (!proxyTarget) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${proxyTarget}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
