import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// The backend must already be running (scripts\run.ps1 or scripts/run.sh) and serving frontend/dist.
// Voice tests feed WAV files as a fake microphone (see fixtures/), found next to this file whatever
// folder npx playwright is started from.
const here = path.dirname(fileURLToPath(import.meta.url))
const fixture = (name) => path.resolve(here, 'fixtures', name)

export default defineConfig({
  testDir: '.',
  globalSetup: './data-guard.js',   // the owner's data is copied aside and put back (data-guard.js)
  timeout: 120_000,
  retries: 0,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'results/report.json' }]],
  outputDir: 'results/artifacts',
  use: {
    baseURL: process.env.AVATAR_URL ?? 'http://127.0.0.1:8000',
    // traces without page snapshots and screenshots: with them, closing the browser after the barge-in
    // test hung for over 2 minutes (the test itself takes 19 s; without tracing the close takes 6 ms).
    // Steps, network, console and sources are still recorded.
    trace: { mode: 'retain-on-failure', screenshots: false, snapshots: false, sources: true },
    screenshot: 'only-on-failure',
    permissions: ['microphone'],
  },
  projects: [
    { name: 'text', testMatch: /phase1_.*\.spec\.js/ },
    {
      // starting the microphone as in a normal browser: no autoplay exemption (the voice project has one,
      // so it never saw what a real browser requires of a tap)
      name: 'mic',
      testMatch: /phase6_mic_start\.spec\.js/,
      use: {
        launchOptions: {
          args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
            `--use-file-for-fake-audio-capture=${fixture('mic_question.wav')}%noloop`],
        },
      },
    },
    {
      // Houston Party Network mode: run against a backend started with --profile hpn
      name: 'hpn',
      testMatch: /(phase7_hpn|load_hpn)\.spec\.js/,
      use: {
        launchOptions: {
          args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
            `--use-file-for-fake-audio-capture=${fixture(process.env.FAKE_MIC ?? 'mic_hpn_wedding.wav')}%noloop`,
            '--autoplay-policy=no-user-gesture-required'],
        },
      },
    },
    {
      name: 'voice',
      testMatch: /(phase[2-5]_.*|phase6_(?!mic_start).*|soak)\.spec\.js/,
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
