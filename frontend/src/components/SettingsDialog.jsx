import { useEffect, useRef, useState } from 'react'
import { isBluetoothMic, listMicrophones, preferredMic, setPreferredMic } from '../audio/microphones.js'

// Persona settings (like Anam's personas / HeyGen LiveAvatar's settings): her name, personality,
// greeting and voice, her memory of the user across sessions (stored on this PC only, erasable), and
// the business she generates leads for. Changes apply from her next reply. Owner link only. A native
// modal <dialog>: Esc, the Close button or a click outside closes it (closedby="any", with a fallback
// for browsers without it).
export default function SettingsDialog({ open, onClose, conv, voice }) {
  const ref = useRef(null)
  const audioRef = useRef(null)
  const p = conv.persona
  const [form, setForm] = useState(null)
  const [saved, setSaved] = useState(false)
  // edited and not saved yet: a persona message arriving now (the page also asks for it when it
  // connects) must not overwrite what is being typed - it did, and Save then stored the old values
  const dirtyRef = useRef(false)
  const [previewing, setPreviewing] = useState(null)
  const [mics, setMics] = useState([])
  const [mic, setMic] = useState(preferredMic())

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) {
      dirtyRef.current = false
      conv.loadPersona()
      setSaved(false)
      listMicrophones().then(setMics)
      dialog.showModal()
    } else if (!open && dialog.open) {
      dialog.close()
    }
  }, [open, conv])

  useEffect(() => {
    if (p && !dirtyRef.current) {
      setForm({ name: p.name, personality: p.personality, greeting: p.greeting, voice: p.voice, memory: p.memory,
        company: p.company ?? '', offer: p.offer ?? '', next_step: p.next_step ?? '', lead_gen: p.lead_gen ?? true })
    }
  }, [p])

  // "Listen": the server sends a short sample of the voice as WAV. Keyed on the stable conv.onJson (conv is
  // a new object every render: it subscribed again on every streamed word)
  const onJson = conv.onJson
  useEffect(() => onJson((msg) => {
    if (msg.type !== 'voice_preview') return
    setPreviewing(null)
    if (!msg.wav_b64) return
    audioRef.current?.pause()
    audioRef.current = new Audio(`data:audio/wav;base64,${msg.wav_b64}`)
    audioRef.current.play().catch(() => {})
  }), [onJson])

  const set = (key) => (e) => {
    setSaved(false)
    dirtyRef.current = true
    setForm((f) => ({ ...f, [key]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }))
  }
  const save = (e) => {
    e.preventDefault()
    conv.savePersona(form)
    dirtyRef.current = false          // the saved values come back from the server
    setSaved(true)
  }
  const preview = () => {
    setPreviewing(form.voice)
    conv.connRef.current?.sendJson({ type: 'preview_voice', voice: form.voice })
  }
  const chooseMic = (e) => {
    setMic(e.target.value)
    setPreferredMic(e.target.value)
    if (voice?.active) voice.restartMic()          // switch now if the microphone is on
  }
  // light dismiss where <dialog closedby> is not supported (e.g. Safari): a click outside the box
  const onClick = (e) => {
    if (e.target !== ref.current) return
    const r = ref.current.getBoundingClientRect()
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) onClose()
  }
  const chosen = mics.find((m) => m.deviceId === mic) ?? mics[0]

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
          <label>Greeting <span className="muted">(when a conversation starts; empty = default)</span>
            <input value={form.greeting} onChange={set('greeting')} maxLength={200}
              placeholder={form.company ? `Hi there! I'm ${form.name} from ${form.company}. What brings you here today?`
                : `Hi there! I'm ${form.name}. Thanks for stopping by! How can I help you today?`}
              data-testid="persona-greeting" />
          </label>
          {p.voices?.length > 0 && (
            <label>Voice <span className="muted">(American English, for customers in the US and Canada)</span>
              <span className="row">
                <select value={form.voice} onChange={set('voice')} data-testid="persona-voice">
                  {p.voices.map((v) => <option key={v} value={v}>{voiceLabel(v)}</option>)}
                </select>
                <button type="button" className="secondary" onClick={preview} disabled={!!previewing}
                  data-testid="voice-preview">{previewing ? 'Loading…' : '▶ Listen'}</button>
              </span>
            </label>
          )}

          <fieldset>
            <legend>Business &amp; leads</legend>
            <label className="check">
              <input type="checkbox" checked={form.lead_gen} onChange={set('lead_gen')} data-testid="lead-gen" />
              Lead generation <span className="muted">(she learns what customers need, suggests the next step and asks for their contact details)</span>
            </label>
            <label>Company name
              <input value={form.company} onChange={set('company')} maxLength={80} placeholder="e.g. Acme Home Solar"
                data-testid="business-company" />
            </label>
            <label>What you offer <span className="muted">(products, services, prices she may mention, areas you serve)</span>
              <textarea value={form.offer} onChange={set('offer')} maxLength={800} rows={3}
                placeholder="e.g. Solar panels for homes in Texas and Florida, with installation and a 25-year warranty."
                data-testid="business-offer" />
            </label>
            <label>Next step for customers
              <input value={form.next_step} onChange={set('next_step')} maxLength={200}
                placeholder="e.g. book a free 15-minute call with our team" data-testid="business-next-step" />
            </label>
          </fieldset>

          {mics.length > 0 && (
            <label>Microphone
              <select value={chosen?.deviceId ?? ''} onChange={chooseMic} data-testid="mic-select">
                {mics.map((m) => <option key={m.deviceId} value={m.deviceId}>{m.label || 'Microphone'}</option>)}
              </select>
              {chosen && isBluetoothMic(chosen.label) && (
                <span className="hint">A Bluetooth headset's microphone switches the headset to phone-call quality: her voice
                  sounds thin and speech recognition gets worse. Pick your computer's built-in microphone and keep the headset for listening.</span>
              )}
            </label>
          )}
          <label className="check">
            <input type="checkbox" checked={form.memory} onChange={set('memory')} data-testid="persona-memory" />
            Remember people between conversations <span className="muted">(a short summary per person, stored only on this PC)</span>
          </label>
          {p.memory && (
            <div className="memory" data-testid="memory">
              <span className="muted">What she remembers about you:</span>{' '}
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

const GENDER = { f: 'female', m: 'male' }
// what the voices sound like (Kokoro's own voice notes and grades); blends average two voices
const NOTES = {
  'af_heart,af_bella': 'warm and upbeat (Heart + Bella)',
  'af_heart,af_nicole': 'warm and soft (Heart + Nicole)',
  af_heart: 'warm, natural',
  af_bella: 'bright, energetic',
  af_nicole: 'soft, calm',
  af_sarah: 'clear, professional',
}

function voiceLabel(v) {
  if (NOTES[v] && v.includes(',')) return `Blend: ${NOTES[v]}`
  const name = v.slice(3)
  const note = NOTES[v] ? ` - ${NOTES[v]}` : ''
  return `${name.charAt(0).toUpperCase()}${name.slice(1)} (${GENDER[v[1]] ?? ''})${note}`
}
