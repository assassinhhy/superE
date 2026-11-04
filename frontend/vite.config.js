import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      // Debug predictions: served by `python -m supere.train.serve` on 8768. Set
      // SUPERE_PREDICT_API to point elsewhere. Keep this entry above "/api".
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
