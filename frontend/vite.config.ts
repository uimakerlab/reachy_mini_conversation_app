import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  build: {
    outDir: "../src/reachy_mini_conversation_app/web-app",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // Python backend (HF secret, status, personalities, camera)
      "/api/config": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "/api/status": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "/api/camera": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      // Daemon endpoints (volume, search, move, state, WebSocket)
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
