import { useCallback, useEffect, useRef, useState } from 'react'
import { Connection } from './connection.js'
import { h264Support } from './h264.js'
import { opusSupport } from './opus.js'

// Conversation state derived only from server events (the server owns the state machine).
export function useConversation() {
  const connRef = useRef(null)
  const [status, setStatus] = useState('idle')
  const [state, setState] = useState('IDLE')
  const [messages, setMessages] = useState([]) // {turnId, role, text, interrupted, source}
  const [behavior, setBehavior] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [notice, setNotice] = useState(null)
  const noticeTimer = useRef(0)
  const [session, setSession] = useState(null)
  const [persona, setPersona] = useState(null)   // name, personality, greeting, voice, memory, voices, remembered
  const [waiting, setWaiting] = useState(null)   // position in line, while another conversation is on
  const [leads, setLeads] = useState(null)       // {leads, csv} (owner link)
  const handlersRef = useRef({ binary: [], json: [] })
  const roleRef = useRef(null)

  const upsertAssistant = useCallback((turnId, fn) => {
    setMessages((prev) => {
      const idx = prev.findLastIndex((m) => m.role === 'assistant' && m.turnId === turnId)
      if (idx === -1) return [...prev, fn({ turnId, role: 'assistant', text: '', interrupted: false, at: Date.now() })]
      const next = prev.slice()
      next[idx] = fn(next[idx])
      return next
    })
  }, [])

  useEffect(() => {
    const conn = new Connection()
    connRef.current = conn
    window.__avatarConn = conn // test hook (Playwright)
    // removed with the connection: a closed Connection's late events (its socket closing) set the status
    // of the page's new one (React StrictMode mounts twice)
    const off = new AbortController()
    const on = (type, fn) => conn.addEventListener(type, fn, { signal: off.signal })
    const showNotice = (text, ms = 4000) => {
      setNotice(text)
      clearTimeout(noticeTimer.current)
      noticeTimer.current = setTimeout(() => setNotice(null), ms)
    }
    on('status', (e) => setStatus(e.detail))
    // the customer's time zone (minutes east of UTC): "what time is it" is answered in their time
    // and the video it can decode: H.264 (~1 Mbit/s) where WebCodecs has it, else JPEG frames (6.7 Mbit/s);
    // the voice as Opus only where this browser really decodes it (opus.js), else PCM
    on('open', async () => {
      const [h264, opus] = await Promise.all([h264Support, opusSupport])
      conn.sendJson({ type: 'hello', voice_out: window.__voiceOut ?? false,
        utc_offset_min: -new Date().getTimezoneOffset(), time_zone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        video: h264 ? ['h264', 'jpeg'] : ['jpeg'], audio: opus ? ['opus', 'pcm'] : ['pcm'], ack: true })
      if (document.hidden) sendVisible()
    })
    // a hidden page gets no video (backend session._set_visible): it piled up unread and the connection
    // was closed every 40 s while the tab was in the background
    const sendVisible = () => conn.sendJson({ type: 'visible', visible: !document.hidden })
    document.addEventListener('visibilitychange', sendVisible)
    on('binary', (e) => handlersRef.current.binary.forEach((h) => h(e.detail)))
    on('json', (e) => {
      const msg = e.detail
      handlersRef.current.json.forEach((h) => h(msg))
      switch (msg.type) {
        case 'session':
          roleRef.current = msg.role
          setSession(msg)
          setWaiting(null)
          // a reconnect starts a new session whose turn ids start again at 1: the earlier messages stay,
          // detached from the ids (the first new reply was appended to the old turn 1's bubble)
          setMessages((prev) => (prev.some((m) => m.turnId != null) ? prev.map((m) => ({ ...m, turnId: null })) : prev))
          break
        case 'waiting':
          setWaiting(msg.position)       // in line behind another conversation (one avatar, one GPU)
          break
        case 'leads':
          setLeads(msg)
          break
        case 'persona':
          setPersona(msg)
          setSession((s) => (s && s.persona !== msg.name ? { ...s, persona: msg.name } : s))
          break
        case 'state':
          setState(msg.state)
          break
        case 'user_message':
          // the greeting answers no message of the user: no "You" bubble
          if (msg.source !== 'greeting') {
            setMessages((prev) => [...prev, { turnId: msg.turn_id, role: 'user', text: msg.text, source: msg.source, at: Date.now() }])
          }
          break
        case 'llm_delta':
          upsertAssistant(msg.turn_id, (m) => ({ ...m, text: m.text + msg.text }))
          break
        case 'assistant_done':
          if (msg.continued) {
            // the user only paused: the next message repeats this part together with the rest
            setMessages((prev) => prev.filter((m) => m.turnId !== msg.turn_id))
            break
          }
          upsertAssistant(msg.turn_id, (m) => ({ ...m, text: msg.interrupted ? msg.text : m.text, interrupted: msg.interrupted, done: true }))
          break
        case 'citations':
          upsertAssistant(msg.turn_id, (m) => ({ ...m, citations: msg.sources }))
          break
        case 'behavior':
          setBehavior(msg)
          break
        case 'transcript':
          // speech was heard but nothing usable came out (noise, a cough, or too unclear)
          if (msg.final && !msg.text && (msg.duration_ms ?? 0) >= 500) showNotice("Sorry, I didn't catch that. Please say it again.")
          break
        case 'error': {
          // a reply that failed, speech recognition that failed, a reply's voice that failed: said in the
          // page (it was ignored - a reply's "…" bubble waited forever). The owner sees the server's reason.
          const text = roleRef.current === 'owner' ? msg.message || 'Something went wrong.'
            : msg.code === 'voice_failed' ? 'The voice failed for this reply.'
              : 'Sorry, something went wrong. Please try again.'
          if (msg.turn_id != null) upsertAssistant(msg.turn_id, (m) => ({ ...m, error: text, done: true }))
          else showNotice(text, 8000)
          break
        }
        case 'metrics':
          setMetrics((prev) => [...prev.slice(-19), msg])
          break
        default:
          break
      }
    })
    conn.connect()
    return () => {
      document.removeEventListener('visibilitychange', sendVisible)
      off.abort()
      conn.close()
      clearTimeout(noticeTimer.current)
    }
  }, [upsertAssistant])

  const sendText = useCallback((text) => connRef.current?.sendJson({ type: 'user_text', text }), [])
  const interrupt = useCallback(() => connRef.current?.sendJson({ type: 'interrupt', reason: 'manual' }), [])
  const loadPersona = useCallback(() => connRef.current?.sendJson({ type: 'get_persona' }), [])
  const savePersona = useCallback((p) => connRef.current?.sendJson({ type: 'set_persona', persona: p }), [])
  const forgetMemory = useCallback(() => connRef.current?.sendJson({ type: 'forget_memory' }), [])
  const loadLeads = useCallback(() => connRef.current?.sendJson({ type: 'get_leads' }), [])
  const onJson = useCallback((h) => {
    handlersRef.current.json.push(h)
    return () => { handlersRef.current.json = handlersRef.current.json.filter((x) => x !== h) }
  }, [])
  const onBinary = useCallback((h) => {
    handlersRef.current.binary.push(h)
    return () => { handlersRef.current.binary = handlersRef.current.binary.filter((x) => x !== h) }
  }, [])

  // no owner tools until the session says so: a customer waiting in line (no session yet) saw Settings
  // and Leads, whose dialogs then stayed on "Loading…"
  return { connRef, status, state, messages, behavior, metrics, session, notice, sendText, interrupt, onJson, onBinary,
    persona, loadPersona, savePersona, forgetMemory, waiting, leads, loadLeads,
    role: session?.role ?? null }
}
