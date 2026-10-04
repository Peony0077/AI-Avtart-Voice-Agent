import { useEffect, useRef, useState } from 'react'

// Persona settings (like Anam's personas / HeyGen LiveAvatar's settings): her name, personality,
// greeting and voice, and her memory of the user across sessions (stored on this PC only, erasable).
// Changes apply from her next reply. A native modal <dialog>: Esc, the Close button or a click
// outside closes it (closedby="any", with a fallback for browsers without it).
export default function SettingsDialog({ open, onClose, conv }) {
  const ref = useRef(null)
  const p = conv.persona
  const [form, setForm] = useState(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      conv.loadPersona()
      setSaved(false)
      dialog.showModal()
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open, conv])

  useEffect(() => {
    if (p) setForm({ name: p.name, personality: p.personality, greeting: p.greeting, voice: p.voice, memory: p.memory })
  }, [p])

  const set = (key) => (e) => {
    setSaved(false)
    setForm((f) => ({ ...f, [key]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }))
  }
  const save = (e) => {
    e.preventDefault()
    conv.savePersona(form)
    setSaved(true)
  }
  // light dismiss where <dialog closedby> is not supported (e.g. Safari): a click outside the box
  const onClick = (e) => {
    if (e.target !== ref.current) return
    const r = ref.current.getBoundingClientRect()
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) onClose()
  }

  return (
    <dialog ref={ref} className="settings" closedby="any" aria-labelledby="settings-title" onClose={onClose}
      onClick={onClick} data-testid="settings">
      {!form ? <p className="muted">Loading…</p> : (
        <form onSubmit={save}>
          <h2 id="settings-title">Avatar settings</h2>
          <label>Name
            <input value={form.name} onChange={set('name')} maxLength={40} required data-testid="persona-name" />
          </label>
          <label>Personality <span className="muted">(how she talks; e.g. "cheerful, a little playful, loves cooking")</span>
            <textarea value={form.personality} onChange={set('personality')} maxLength={600} rows={3}
              data-testid="persona-personality" />
          </label>
          <label>Greeting <span className="muted">(when you press Start voice; empty = default)</span>
            <input value={form.greeting} onChange={set('greeting')} maxLength={200}
              placeholder={`Hi, I'm ${form.name}! How can I help you today?`} data-testid="persona-greeting" />
          </label>
          {p.voices?.length > 0 && (
            <label>Voice
              <select value={form.voice} onChange={set('voice')} data-testid="persona-voice">
                {p.voices.map((v) => <option key={v} value={v}>{voiceLabel(v)}</option>)}
              </select>
            </label>
          )}
          <label className="check">
            <input type="checkbox" checked={form.memory} onChange={set('memory')} data-testid="persona-memory" />
            Remember me between conversations <span className="muted">(a short summary, stored only on this PC)</span>
          </label>
          {p.memory && (
            <div className="memory" data-testid="memory">
              <span className="muted">What she remembers:</span>{' '}
              {p.remembered ? <q>{p.remembered}</q> : <span className="muted">nothing yet</span>}
              {p.remembered && (
                <button type="button" className="secondary" onClick={conv.forgetMemory} data-testid="forget">Forget me</button>
              )}
            </div>
          )}
          <div className="actions">
            {saved && <span className="muted" role="status">Saved: applies from her next reply.</span>}
            <button type="button" className="secondary" onClick={onClose}>Close</button>
            <button type="submit" data-testid="persona-save">Save</button>
          </div>
        </form>
      )}
    </dialog>
  )
}

const ACCENT = { a: 'American', b: 'British' }
const GENDER = { f: 'female', m: 'male' }

function voiceLabel(v) {
  const name = v.slice(3)
  return `${name.charAt(0).toUpperCase()}${name.slice(1)} (${ACCENT[v[0]] ?? ''} ${GENDER[v[1]] ?? ''})`
}
