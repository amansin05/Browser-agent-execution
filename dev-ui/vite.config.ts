import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React dev server runs on 5173 and talks to the Python backend over a WebSocket
// at ws://localhost:8000/ws (overridable with VITE_WS_URL).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
