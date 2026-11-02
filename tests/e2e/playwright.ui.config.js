import { defineConfig } from '@playwright/test'

// Isolated browser checks: fixture WebSockets only, no GPU/backend or owner-data guard.
export default defineConfig({
  testDir: './ui',
  timeout: 25_000,
  workers: 1,
  reporter: 'list',
  outputDir: 'results/ui',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 1440, height: 1050 },
    screenshot: 'only-on-failure',
    launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] },
  },
  webServer: {
    command: 'npm --prefix ../../frontend run dev -- --host 127.0.0.1',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: true,
  },
})
