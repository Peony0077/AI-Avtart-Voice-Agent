// The owner's own data (memory, persona, leads, customers' memories under data\) must not change when
// the tests run against the owner's server: test conversations went into her memory of the owner
// ("...or your bakery website") and one answer refused to tell the time because of it. Before the run
// it is copied aside and the memory cleared (each run starts the same way); after the run it is put
// back exactly. Only when the server runs on this PC.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { withOwnerToken } from './helpers.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const DATA = path.resolve(here, '../../data')
const BACKUP = path.resolve(here, 'results/.owner-data')
const FINAL = path.resolve(here, 'results/.owner-data-final')   // the teardown's copy of BACKUP
const FILES = ['memory.json', 'persona.json', 'leads.jsonl']
export const LOCAL = !process.env.AVATAR_URL || /\/\/(127\.0\.0\.1|localhost)[:/]/.test(process.env.AVATAR_URL)

// A run that was interrupted (Ctrl+C, a crash) before its teardown leaves the owner's real data only in
// the backup - data\ then holds the test run's. So an existing backup is put back first; it is never
// deleted unread, and an incomplete one stops the run (the owner checks it by hand).
const isWhole = (dir) => {
  try {
    const known = JSON.parse(fs.readFileSync(path.join(dir, 'visitors.json'), 'utf8'))   // written last
    return known.every((f) => fs.existsSync(path.join(dir, `visitor-${f}`)))
  } catch {
    return false
  }
}

function recoverInterruptedRun() {
  if (!isWhole(BACKUP) && isWhole(FINAL)) {
    // interrupted in the teardown: FINAL is its complete copy of the backup (BACKUP already put back and
    // removed, or half re-copied)
    fs.rmSync(`${BACKUP}.tmp`, { recursive: true, force: true })
    fs.cpSync(FINAL, `${BACKUP}.tmp`, { recursive: true })
    fs.rmSync(BACKUP, { recursive: true, force: true })
    fs.renameSync(`${BACKUP}.tmp`, BACKUP)
  }
  if (fs.existsSync(BACKUP)) {
    if (!isWhole(BACKUP)) {
      throw new Error(`data-guard: ${BACKUP} is an incomplete backup of the owner's data from an interrupted `
        + 'test run. Compare it with data\\, put back what is missing by hand, then delete it. Not starting.')
    }
    console.warn(`data-guard: putting back the owner's data from an interrupted test run (${BACKUP})`)
    restore()
  }
  fs.rmSync(FINAL, { recursive: true, force: true })
}

export function guard() {
  if (!LOCAL) return
  recoverInterruptedRun()
  // copied into a temporary folder and renamed when complete, visitors.json last: a backup folder is
  // always a whole one (a leftover .tmp is an unfinished copy, made while data\ was still untouched)
  const tmp = `${BACKUP}.tmp`
  fs.rmSync(tmp, { recursive: true, force: true })
  fs.mkdirSync(tmp, { recursive: true })
  for (const f of FILES) {
    if (fs.existsSync(path.join(DATA, f))) fs.copyFileSync(path.join(DATA, f), path.join(tmp, f))
  }
  const visitors = path.join(DATA, 'visitors')
  const known = fs.existsSync(visitors) ? fs.readdirSync(visitors) : []
  for (const f of known) fs.copyFileSync(path.join(visitors, f), path.join(tmp, `visitor-${f}`))
  fs.writeFileSync(path.join(tmp, 'visitors.json'), JSON.stringify(known))
  fs.renameSync(tmp, BACKUP)
  fs.rmSync(path.join(DATA, 'memory.json'), { force: true })
}

export function restore() {
  if (!LOCAL || !fs.existsSync(BACKUP)) return
  for (const f of FILES) {
    const saved = path.join(BACKUP, f)
    if (fs.existsSync(saved)) fs.copyFileSync(saved, path.join(DATA, f))
    else fs.rmSync(path.join(DATA, f), { force: true })
  }
  const visitors = path.join(DATA, 'visitors')
  const known = JSON.parse(fs.readFileSync(path.join(BACKUP, 'visitors.json'), 'utf8'))
  if (fs.existsSync(visitors)) {
    for (const f of fs.readdirSync(visitors)) if (!known.includes(f)) fs.rmSync(path.join(visitors, f))
  }
  if (known.length) fs.mkdirSync(visitors, { recursive: true })
  for (const f of known) fs.copyFileSync(path.join(BACKUP, `visitor-${f}`), path.join(visitors, f))
  fs.rmSync(BACKUP, { recursive: true, force: true })
}

// The server merges memories and writes leads in the background after a conversation, and only while
// no conversation needs the LLM (services.py) - so the last test's update can land after the restore
// (seen: the owner's memory rewritten 2 minutes after the run). Keep restoring until nothing has
// changed for 20 s (at most 2 minutes).
const snapshot = () => [...FILES.map((f) => path.join(DATA, f)), ...(fs.existsSync(path.join(DATA, 'visitors'))
  ? fs.readdirSync(path.join(DATA, 'visitors')).map((f) => path.join(DATA, 'visitors', f)) : [])]
  .map((p) => (fs.existsSync(p) ? `${p}:${fs.statSync(p).mtimeMs}` : `${p}:-`)).join('|')

// Wait until the server has no conversation and no memory / lead job left (GET /health), so the last
// test's background writes land before the owner's data is put back. A time window alone was not
// enough: after a 10-minute soak the jobs (a long transcript) finished after it and stayed.
async function serverIdle(maxMs = 300_000) {
  const base = process.env.AVATAR_URL ?? 'http://127.0.0.1:8000'
  let idleFor = 0
  for (let waited = 0; waited < maxMs && idleFor < 5_000; waited += 1_000) {
    await new Promise((r) => setTimeout(r, 1_000))
    try {
      const h = await (await fetch(`${base}${withOwnerToken('/health')}`)).json()
      idleFor = !h.active_session && !h.background_jobs ? idleFor + 1_000 : 0
    } catch {
      return                            // no server (stopped): nothing more will be written
    }
  }
}

export default async function globalSetup() {
  guard()
  return async () => {                  // global teardown
    if (!LOCAL) return
    await serverIdle()
    const saved = FINAL
    fs.rmSync(saved, { recursive: true, force: true })
    fs.cpSync(BACKUP, saved, { recursive: true })
    restore()
    let last = snapshot()
    let quiet = 0
    for (let waited = 0; waited < 120_000 && quiet < 20_000; waited += 2_000) {
      await new Promise((r) => setTimeout(r, 2_000))
      const now = snapshot()
      if (now !== last) {
        fs.cpSync(saved, BACKUP, { recursive: true })
        restore()                        // a late background write: put the owner's data back again
        last = snapshot()
        quiet = 0
      } else {
        quiet += 2_000
      }
    }
    fs.rmSync(saved, { recursive: true, force: true })
  }
}
