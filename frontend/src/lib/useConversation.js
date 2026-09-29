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
  const [session, setSession] = useState(null)
  const handlersRef = useRef({ binary: [], json: [] })

  const upsertAssistant = useCallback((turnId, fn) => {
    setMessages((prev) => {
      const idx = prev.findLastIndex((m) => m.role === 'assistant' && m.turnId === turnId)
      if (idx === -1) return [...prev, fn({ turnId, role: 'assistant', text: '', interrupted: false })]
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
    conn.addEventListener('open', () => conn.sendJson({ type: 'hello', voice_out: window.__voiceOut ?? false }))
    conn.addEventListener('binary', (e) => handlersRef.current.binary.forEach((h) => h(e.detail)))
    conn.addEventListener('json', (e) => {
      const msg = e.detail
      handlersRef.current.json.forEach((h) => h(msg))
      switch (msg.type) {
        case 'session':
          setSession(msg)
          break
        case 'state':
          setState(msg.state)
          break
        case 'user_message':
          setMessages((prev) => [...prev, { turnId: msg.turn_id, role: 'user', text: msg.text, source: msg.source }])
          break
        case 'llm_delta':
          upsertAssistant(msg.turn_id, (m) => ({ ...m, text: m.text + msg.text }))
          break
        case 'assistant_done':
          upsertAssistant(msg.turn_id, (m) => ({ ...m, text: msg.interrupted ? msg.text : m.text, interrupted: msg.interrupted, done: true }))
          break
        case 'behavior':
          setBehavior(msg)
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
  const onJson = useCallback((h) => {
    handlersRef.current.json.push(h)
    return () => { handlersRef.current.json = handlersRef.current.json.filter((x) => x !== h) }
  }, [])
  const onBinary = useCallback((h) => {
    handlersRef.current.binary.push(h)
    return () => { handlersRef.current.binary = handlersRef.current.binary.filter((x) => x !== h) }
  }, [])

  return { connRef, status, state, messages, behavior, metrics, session, sendText, interrupt, onJson, onBinary }
}
