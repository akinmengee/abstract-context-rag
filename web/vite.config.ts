import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // 0.0.0.0 so the app is reachable from a phone on the same network.
    host: true,
    port: 5173,
  },
});
