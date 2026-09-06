import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В dev проксируем /api на локальный бэкенд (uvicorn на :8000).
// В проде статику и /api раздаёт Caddy, прокси не используется.
const apiTarget =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
