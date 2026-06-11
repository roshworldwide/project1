import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Served from the root of the Python server, so absolute paths are correct
  // and survive refreshes on deep client routes like /runs/<id>.
  base: "/",
  build: {
    outDir: "../src/holdout/dashboard_dist",
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 8192,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:4517",
    },
  },
});
