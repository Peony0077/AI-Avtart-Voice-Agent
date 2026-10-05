import { useCallback, useEffect, useRef, useState } from 'react'
import { Connection } from './connection.js'

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
    conn.addEventListener('status', (e) => setStatus(e.detail))
    // the customer's time zone (minutes east of UTC): "what time is it" is answered in their time
    conn.addEventListener('open', () => conn.sendJson({ type: 'hello', voice_out: window.__voiceOut ?? false,
      utc_offset_min: -new Date().getTimezoneOffset(), time_zone: Intl.DateTimeFormat().resolvedOptions().timeZone }))
    conn.addEventListener('binary', (e) => handlersRef.current.binary.forEach((h) => h(e.detail)))
    conn.addEventListener('json', (e) => {
      const msg = e.detail
      handlersRef.current.json.forEach((h) => h(msg))
      switch (msg.type) {
        case 'session':
          setSession(msg)
          setWaiting(null)
          break
        case 'waiting':
          setWaiting(msg.position)       // in line behind another conversation (one avatar, one GPU)
          break
        case 'leads':
          setLeads(msg)
          break
        case 'persona':
          setPersona(msg)
          setSession((s) => (s ? { ...s, persona: msg.name } : s))
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
          if (msg.final && !msg.text && (msg.duration_ms ?? 0) >= 500) {
            setNotice("Sorry, I didn't catch that. Please say it again.")
            clearTimeout(noticeTimer.current)
            noticeTimer.current = setTimeout(() => setNotice(null), 4000)
          }
          break
        case 'metrics':
          setMetrics((prev) => [...prev.slice(-19), msg])
          break
        default:
          break
      }
    })
    conn.connect()
    return () => conn.close()
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

  return { connRef, status, state, messages, behavior, metrics, session, notice, sendText, interrupt, onJson, onBinary,
    persona, loadPersona, savePersona, forgetMemory, waiting, leads, loadLeads,
    role: session?.role ?? 'owner' }
}
