import { defineConfig } from '@playwright/test'
import path from 'node:path'

// The backend must already be running (scripts\run.ps1) and serving frontend/dist.
// Voice tests feed WAV files as a fake microphone (see fixtures/).
const fixture = (name) => path.resolve('fixtures', name)

export default defineConfig({
  testDir: '.',
  timeout: 120_000,
  retries: 0,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'results/report.json' }]],
  outputDir: 'results/artifacts',
  use: {
    baseURL: process.env.AVATAR_URL ?? 'http://127.0.0.1:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    permissions: ['microphone'],
  },
  projects: [
    { name: 'text', testMatch: /phase1_.*\.spec\.js/ },
    {
      name: 'voice',
      testMatch: /(phase[2-6]_.*|soak)\.spec\.js/,
      use: {
        launchOptions: {
          args: [
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
            `--use-file-for-fake-audio-capture=${fixture(process.env.FAKE_MIC ?? 'mic_question.wav')}${process.env.FAKE_MIC_LOOP ? '' : '%noloop'}`,
            '--autoplay-policy=no-user-gesture-required',
          ],
        },
      },
    },
  ],
})
