import type { NextConfig } from "next";

const apiOrigin = process.env.ECOVISION_API_ORIGIN ?? "http://127.0.0.1:8000";
const staticExport = process.env.ECOVISION_STATIC_EXPORT === "true";
const pagesBasePath = staticExport ? process.env.PAGES_BASE_PATH ?? "" : "";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: staticExport ? "export" : undefined,
  basePath: pagesBasePath,
  trailingSlash: staticExport,
  images: {
    unoptimized: staticExport
  },
  env: {
    NEXT_PUBLIC_STATIC_DEMO: staticExport ? "true" : "false"
  },
  ...(staticExport
    ? {}
    : {
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: `${apiOrigin}/api/:path*`
            }
          ];
        }
      })
};

export default nextConfig;
