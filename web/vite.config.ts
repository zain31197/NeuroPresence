import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// While developing, the pages come from Vite and every /api call (including the
// live stream) is passed on to the Python backend. Set NP_API if it runs elsewhere.
const backend = process.env.NP_API ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { '/api': { target: backend, ws: true } },
  },
})
