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
      // fastrtc WebRTC signaling
      "/webrtc": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      // SSE events stream
      "/api/events": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      // Python backend REST endpoints
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
      "/api/personalities": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "/api/voices": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "/api/validate_api_key": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      "/api/openai_api_key": {
        target: "http://localhost:7860",
        changeOrigin: true,
      },
      // Daemon endpoints (volume, search, move, state)
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
