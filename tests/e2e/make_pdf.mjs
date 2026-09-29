// Print an HTML file to PDF with Playwright's Chromium (used for RAG test fixtures).
// usage: node make_pdf.mjs input.html output.pdf
import { chromium } from '@playwright/test'
import { pathToFileURL } from 'node:url'

const [input, output] = process.argv.slice(2)
const browser = await chromium.launch()
const page = await browser.newPage()
await page.goto(pathToFileURL(input).href)
await page.pdf({ path: output, format: 'A4' })
await browser.close()
console.log(`pdf written: ${output}`)
