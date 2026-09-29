import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // 0.0.0.0 so the app is reachable from a phone on the same network.
    host: true,
    port: 5173,
    proxy: {
      // Mirrors nginx.conf's /api/ proxy in the Docker image, so the client
      // can use plain relative paths in both dev and production - no
      // backend URL baked into the build, no CORS to configure either way.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
