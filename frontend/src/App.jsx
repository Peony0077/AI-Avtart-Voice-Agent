import { useEffect, useState } from 'react'

// Phase 0 scaffold: shows backend health. Replaced by the conversation UI in Phase 1.
export default function App() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch('/health/gpu').then((r) => r.json()).then(setHealth).catch(() => setHealth({ error: true }))
  }, [])

  return (
    <main className="app">
      <h1>AI Avatar</h1>
      <pre data-testid="health">{health ? JSON.stringify(health, null, 2) : 'checking backend…'}</pre>
    </main>
  )
}
