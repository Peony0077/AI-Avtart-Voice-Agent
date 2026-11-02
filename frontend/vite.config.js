import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { copyFileSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

// StaticFiles serves directory index files, so the branded entry works on the existing
// backend and tunnel as soon as a new frontend build is written. Assets and /ws stay at root.
function brandedEntry() {
  let outDir
  return {
    name: 'hpn-branded-entry',
    apply: 'build',
    configResolved(config) { outDir = resolve(config.root, config.build.outDir) },
    closeBundle() {
      const brandedDir = resolve(outDir, 'houston-party-network')
      mkdirSync(brandedDir, { recursive: true })
      copyFileSync(resolve(outDir, 'index.html'), resolve(brandedDir, 'index.html'))
    },
  }
}

// Dev server proxies API + WebSocket to the FastAPI backend on :8000.
// In production the backend serves the built files from frontend/dist (one origin).
export default defineConfig({
  plugins: [react(), brandedEntry()],
  server: {
    port: 5173,
    proxy: {
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
      '/health': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
