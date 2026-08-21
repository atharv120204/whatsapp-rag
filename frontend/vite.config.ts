import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Everything under /api goes to the FastAPI backend, so the frontend
      // has no hardcoded host and CORS stays out of the way in development.
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
