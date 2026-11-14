import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      // The desk calls 8769 directly. This proxy remains for anything still using a relative news path.
      "/api/news": "http://127.0.0.1:8769",
      "/api/predict": process.env.SUPERE_PREDICT_API || "http://127.0.0.1:8768",
      "/api": "http://127.0.0.1:8766",
      "/ws": { target: "ws://127.0.0.1:8766", ws: true },
    },
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: false,
  },
});
