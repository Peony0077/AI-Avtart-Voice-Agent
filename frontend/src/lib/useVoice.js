import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioEngine } from '../audio/AudioEngine.js'
import { KIND_AUDIO, parsePacket } from './protocol.js'

// Voice: server audio → player (also for typed messages: the "speaker"), playback acks → server,
// and optionally the microphone → server (binary PCM16 16 kHz).
// Client-side stale-packet guard: only audio of the newest turn is played; after an
// interrupt, everything up to that turn id is rejected.
export function useVoice(conv) {
  const engineRef = useRef(null)
  const turnRef = useRef(0)           // newest turn announced by the server
  const rejectUpToRef = useRef(0)     // turns <= this id are stale (interrupted)
  const mutedRef = useRef(false)
  const [active, setActive] = useState(false)      // microphone on
  const [speaker, setSpeaker] = useState(false)    // assistant replies are spoken
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const [speechDetected, setSpeechDetected] = useState(false)
  const [error, setError] = useState(null)
  const statsRef = useRef({ interrupts: [], blockedStale: 0, pushes: {}, gaps: {} })

  const send = (msg) => conv.connRef.current?.sendJson(msg)

  useEffect(() => {
    const offJson = conv.onJson((msg) => {
      const engine = engineRef.current
      switch (msg.type) {
        case 'user_message':
          turnRef.current = msg.turn_id
          break
        case 'interrupt':
          rejectUpToRef.current = Math.max(rejectUpToRef.current, msg.turn_id)
          statsRef.current.interrupts.push({ turnId: msg.turn_id, reason: msg.reason, receivedAt: performance.now() })
          engine?.flush()
          break
        case 'tts_done':
          if (msg.turn_id === turnRef.current) engine?.endTurn(msg.turn_id)
          break
        case 'vad':
          setSpeechDetected(msg.event === 'start')
          break
        case 'session':
          // (re)connected: restore speaker and microphone on the server
          if (engine && !mutedRef.current) send({ type: 'speaker', enabled: true })
          if (engine?.micOn) send({ type: 'voice', enabled: true, mic: engine.micInfo })
          break
        default:
          break
      }
    })
    const offBinary = conv.onBinary((buffer) => {
      const packet = parsePacket(buffer)
      if (packet.kind !== KIND_AUDIO) return
      if (packet.turnId <= rejectUpToRef.current || packet.turnId !== turnRef.current) {
        statsRef.current.blockedStale++
        return
      }
      statsRef.current.pushes[packet.turnId] = (statsRef.current.pushes[packet.turnId] ?? 0) + 1
      engineRef.current?.push(packet.turnId, packet.payload)
    })
    return () => {
      offJson()
      offBinary()
    }
  }, [conv])

  useEffect(() => {
    if (!active) return undefined
    const id = setInterval(() => setLevel(engineRef.current?.level ?? 0), 100)
    return () => clearInterval(id)
  }, [active])

  // Create the audio engine. Must run inside a click/submit handler, before any await.
  const engine = () => {
    if (engineRef.current) return engineRef.current
    const e = new AudioEngine()
    e.addEventListener('player', (ev) => {
      const m = ev.detail
      if (m.type === 'flushed') {
        const last = statsRef.current.interrupts.at(-1)
        if (last && last.flushedAt === undefined) last.flushedAt = performance.now()
      }
      if (m.type === 'started' || m.type === 'progress' || m.type === 'ended') {
        send({ type: 'playback', turn_id: m.turnId, event: m.type, played_ms: Math.round(m.playedMs ?? 0),
          gap_ms: Math.round(m.gapMs ?? 0), client_time: Date.now() })
        if (m.type === 'ended') statsRef.current.gaps[m.turnId] = Math.round(m.gapMs ?? 0)
      }
    })
    engineRef.current = e
    window.__audioEngine = e // test hooks
    window.__voiceStats = statsRef.current
    return e
  }

  // Typed messages: speak the reply too (avatar lip sync needs the audio). Called from the Send handler.
  const ensureSpeaker = useCallback(() => {
    if (mutedRef.current || conv.session?.voice_available === false) return
    if (engineRef.current?.playCtx) return
    const e = engine()
    send({ type: 'speaker', enabled: true })     // before the text message: the server speaks this turn
    setSpeaker(true)
    e.startPlayback().catch((err) => setError(err.message ?? String(err)))
  }, [conv])

  const toggleMute = useCallback(() => {
    const next = !mutedRef.current
    mutedRef.current = next
    setMuted(next)
    if (next) engineRef.current?.flush()
    send({ type: 'speaker', enabled: !next })
  }, [conv])

  const start = useCallback(async () => {
    setError(null)
    try {
      const e = engine()
      const playback = e.startPlayback()               // AudioContext created inside the click
      await playback
      await e.startMic((pcm) => conv.connRef.current?.sendBinary(pcm))
      send({ type: 'voice', enabled: true, mic: e.micInfo })
      if (!mutedRef.current) send({ type: 'speaker', enabled: true })
      send({ type: 'greet' })     // she says hello (once a session, only before anything else was said)
      setSpeaker(true)
      setActive(true)
    } catch (err) {
      setError(err.message ?? String(err))
    }
  }, [conv])

  // Stop voice: microphone off; typed messages are still spoken (Mute turns that off).
  const stop = useCallback(async () => {
    send({ type: 'voice', enabled: false })
    await engineRef.current?.stopMic()
    setActive(false)
    setSpeechDetected(false)
  }, [conv])

  return { active, speaker, muted, level, speechDetected, error, start, stop, ensureSpeaker, toggleMute, engineRef }
}
