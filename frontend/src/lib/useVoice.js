import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioEngine } from '../audio/AudioEngine.js'
import { KIND_AUDIO, parsePacket } from './protocol.js'

// Voice mode: mic → server (binary PCM16 16 kHz), server audio → player, playback acks → server.
// Client-side stale-packet guard: only audio of the newest turn is played; after an
// interrupt, everything up to that turn id is rejected.
export function useVoice(conv) {
  const engineRef = useRef(null)
  const turnRef = useRef(0)           // newest turn announced by the server
  const rejectUpToRef = useRef(0)     // turns <= this id are stale (interrupted)
  const [active, setActive] = useState(false)
  const [level, setLevel] = useState(0)
  const [speechDetected, setSpeechDetected] = useState(false)
  const [error, setError] = useState(null)
  const statsRef = useRef({ interrupts: [], blockedStale: 0, pushes: {} })

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
          // (re)connected: re-enable voice on the server if it was on
          if (engineRef.current) send({ type: 'voice', enabled: true })
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

  const start = useCallback(async () => {
    setError(null)
    try {
      const engine = new AudioEngine()
      engine.addEventListener('player', (e) => {
        const m = e.detail
        if (m.type === 'flushed') {
          const last = statsRef.current.interrupts.at(-1)
          if (last && last.flushedAt === undefined) last.flushedAt = performance.now()
        }
        if (m.type === 'started' || m.type === 'progress' || m.type === 'ended') {
          send({ type: 'playback', turn_id: m.turnId, event: m.type, played_ms: Math.round(m.playedMs ?? 0), client_time: Date.now() })
        }
      })
      await engine.start((pcm) => conv.connRef.current?.sendBinary(pcm))
      engineRef.current = engine
      window.__audioEngine = engine // test hooks
      window.__voiceStats = statsRef.current
      send({ type: 'voice', enabled: true })
      setActive(true)
    } catch (err) {
      setError(err.message ?? String(err))
    }
  }, [conv])

  const stop = useCallback(async () => {
    send({ type: 'voice', enabled: false })
    await engineRef.current?.stop()
    engineRef.current = null
    setActive(false)
    setSpeechDetected(false)
  }, [conv])

  return { active, level, speechDetected, error, start, stop, engineRef }
}
