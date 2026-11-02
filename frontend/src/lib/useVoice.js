import { useCallback, useEffect, useRef, useState } from 'react'
import { AudioEngine } from '../audio/AudioEngine.js'
import { betterThanBluetooth, setPreferredMic } from '../audio/microphones.js'
import { pcm16ToFloat } from '../audio/AudioEngine.js'
import { decodeOpus } from './opus.js'
import { KIND_AUDIO, KIND_AUDIO_OPUS, packetKind, parsePacket } from './protocol.js'

// Voice: server audio → player (also for typed messages: the "speaker"), playback acks → server,
// and optionally the microphone → server (binary PCM16 16 kHz).
// Client-side stale-packet guard: only audio of the newest turn is played; after an
// interrupt, everything up to that turn id is rejected.
export function useVoice(conv) {
  const engineRef = useRef(null)
  const turnRef = useRef(0)           // newest turn announced by the server
  const rejectUpToRef = useRef(0)     // turns <= this id are stale (interrupted)
  const mutedRef = useRef(false)
  const speakerOnRef = useRef(false)  // the server was told to speak replies (Speaker on)
  const [active, setActive] = useState(false)      // microphone on
  const [speaker, setSpeaker] = useState(false)    // assistant replies are spoken
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const [speechDetected, setSpeechDetected] = useState(false)
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)   // waiting for the microphone (permission prompt)
  const [micHint, setMicHint] = useState(null)      // {label, deviceId}: a better microphone than a Bluetooth headset's
  const silentTimerRef = useRef(null)
  const statsRef = useRef({ interrupts: [], blockedStale: 0, pushes: {}, gaps: {} })
  // A reply's first audio waits startHoldRef ms (set by AvatarStage from how late the lip-synced frames
  // arrived): { turnId, packets, end, timer } while it waits
  const startHoldRef = useRef(0)
  const holdRef = useRef(null)
  const lastTurnRef = useRef(0)
  const chainRef = useRef(Promise.resolve())   // audio in arrival order (Opus decodes are asynchronous)

  const send = (msg) => conv.connRef.current?.sendJson(msg)

  useEffect(() => {
    const dropHold = () => {
      clearTimeout(holdRef.current?.timer)
      holdRef.current = null
    }
    const releaseHold = () => {
      const h = holdRef.current
      dropHold()
      if (!h || mutedRef.current || h.turnId <= rejectUpToRef.current || h.turnId !== turnRef.current) return
      for (const samples of h.packets) engineRef.current?.pushFloat(h.turnId, samples)
      if (h.end) engineRef.current?.endTurn(h.turnId)
    }
    // a decoded clause to the player (or held: the reply's start hold)
    const deliver = (turnId, samples) => {
      if (mutedRef.current) return                       // muted: audio already on its way is not played
      if (turnId <= rejectUpToRef.current || turnId !== turnRef.current) {   // interrupted while decoding
        statsRef.current.blockedStale++
        return
      }
      const h = holdRef.current
      if (h && h.turnId === turnId) {
        h.packets.push(samples)
        return
      }
      if (turnId !== lastTurnRef.current) {             // the first audio of a reply
        lastTurnRef.current = turnId
        if (startHoldRef.current > 0) {
          dropHold()
          holdRef.current = { turnId, packets: [samples], end: false, timer: setTimeout(releaseHold, startHoldRef.current) }
          statsRef.current.holds = [...(statsRef.current.holds ?? []), startHoldRef.current].slice(-50)
          return
        }
      }
      engineRef.current?.pushFloat(turnId, samples)
    }
    const inOrder = (step) => { chainRef.current = chainRef.current.then(step).catch(() => {}) }
    const offJson = conv.onJson((msg) => {
      const engine = engineRef.current
      switch (msg.type) {
        case 'user_message':
          turnRef.current = msg.turn_id
          break
        case 'interrupt':
          rejectUpToRef.current = Math.max(rejectUpToRef.current, msg.turn_id)
          statsRef.current.interrupts.push({ turnId: msg.turn_id, reason: msg.reason, receivedAt: performance.now() })
          if (statsRef.current.interrupts.length > 50) statsRef.current.interrupts.shift()
          if (holdRef.current && holdRef.current.turnId <= msg.turn_id) dropHold()
          engine?.flush()
          break
        case 'tts_done':
          inOrder(() => {                                // after the reply's last clause is decoded
            if (holdRef.current?.turnId === msg.turn_id) holdRef.current.end = true
            else if (msg.turn_id === turnRef.current) engineRef.current?.endTurn(msg.turn_id)
          })
          break
        case 'vad':
          setSpeechDetected(msg.event === 'start')
          break
        case 'session':
          dropHold()
          // turn ids start again: what the old session interrupted or played says nothing about the new one
          lastTurnRef.current = 0
          turnRef.current = 0
          rejectUpToRef.current = 0
          engine?.resetTurns()
          // (re)connected: restore speaker and microphone on the server
          if (engine) send({ type: 'speaker', enabled: !mutedRef.current })   // Mute holds over a reconnect
          if (engine?.micOn) send({ type: 'voice', enabled: true, mic: engine.micInfo })
          break
        default:
          break
      }
    })
    const offBinary = conv.onBinary((buffer) => {
      const kind = packetKind(buffer)
      if (kind !== KIND_AUDIO && kind !== KIND_AUDIO_OPUS) return        // video: not copied for nothing
      const packet = parsePacket(buffer)
      if (packet.turnId <= rejectUpToRef.current || packet.turnId !== turnRef.current) {
        statsRef.current.blockedStale++
        return
      }
      countFor(statsRef.current.pushes, packet.turnId, (statsRef.current.pushes[packet.turnId] ?? 0) + 1)
      // decoding starts at once; the clauses reach the player in the order they came
      const samples = packet.kind === KIND_AUDIO_OPUS ? decodeOpus(packet.payload) : pcm16ToFloat(packet.payload)
      inOrder(async () => deliver(packet.turnId, await samples))
    })
    return () => {
      offJson()
      offBinary()
      dropHold()
    }
    // stable: conv is a new object every render - keyed on it, every render (a streamed word)
    // subscribed again and dropped the reply's held first audio
  }, [conv.onJson, conv.onBinary])

  useEffect(() => {
    if (!active) return undefined
    const id = setInterval(() => setLevel(engineRef.current?.level ?? 0), 100)
    return () => clearInterval(id)
  }, [active])

  // A phone lock, a call or another app suspends the playback AudioContext (iOS: "interrupted"); nothing
  // resumed it, so every later reply was silent - and the avatar never lip-synced - until a reload.
  useEffect(() => {
    const wake = () => {
      const ctx = engineRef.current?.playCtx
      if (document.visibilityState === 'visible' && ctx && ctx.state !== 'running' && ctx.state !== 'closed') {
        ctx.resume().catch(() => {})
      }
    }
    document.addEventListener('visibilitychange', wake)
    window.addEventListener('pageshow', wake)
    window.addEventListener('pointerdown', wake)     // a tap is a user gesture: always enough to resume
    return () => {
      document.removeEventListener('visibilitychange', wake)
      window.removeEventListener('pageshow', wake)
      window.removeEventListener('pointerdown', wake)
    }
  }, [])

  // the page goes away: microphone and AudioContexts off (they kept the microphone in use)
  useEffect(() => () => {
    clearTimeout(silentTimerRef.current)
    const e = engineRef.current
    engineRef.current = null
    e?.stop().catch(() => {})
  }, [])

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
        if (m.type === 'ended') countFor(statsRef.current.gaps, m.turnId, Math.round(m.gapMs ?? 0))
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
    const e = engine()
    if (speakerOnRef.current && e.playCtx) {
      if (e.playCtx.state !== 'running') e.playCtx.resume().catch(() => {})   // inside the Send click
      return
    }
    // (not "playCtx exists": a microphone that was refused left playback made but the speaker off, and
    // every typed reply came back silent)
    speakerOnRef.current = true
    send({ type: 'speaker', enabled: true })     // before the text message: the server speaks this turn
    setSpeaker(true)
    e.startPlayback().catch((err) => { setError(err.message ?? String(err)); report('speaker_start', err) })
  }, [conv])

  const toggleMute = useCallback(() => {
    const next = !mutedRef.current
    mutedRef.current = next
    setMuted(next)
    if (next) {
      engineRef.current?.flush()
      clearTimeout(holdRef.current?.timer)     // a reply's held first audio is not played either
      holdRef.current = null
    }
    send({ type: 'speaker', enabled: !next })
  }, [conv])

  // Problems the user sees are reported to the server log too: the owner's microphone never reached
  // the server, and the page kept the reason to itself.
  const report = (where, err) => send({ type: 'client_error', where, name: err?.name ?? 'Error',
    message: err?.detail || err?.message || String(err), userAgent: navigator.userAgent })

  const start = useCallback(async () => {
    setError(null)
    setStarting(true)
    try {
      const e = engine()
      // both AudioContexts and the permission prompt start inside the click, before any await
      const playback = e.startPlayback()
      const mic = e.startMic((pcm) => conv.connRef.current?.sendBinary(pcm))
      const [p, m] = await Promise.allSettled([playback, mic])
      if (m.status === 'rejected') throw m.reason
      if (p.status === 'rejected') {
        await e.stopMic()
        throw p.reason
      }
      send({ type: 'voice', enabled: true, mic: e.micInfo })
      if (!mutedRef.current) send({ type: 'speaker', enabled: true })
      speakerOnRef.current = true
      send({ type: 'greet' })     // she says hello (once a session, only before anything else was said)
      setSpeaker(true)
      setActive(true)
      betterThanBluetooth(e.micInfo?.device).then(setMicHint)
      // a muted or disabled microphone sends exact zeros: say so instead of "listening" to nothing
      clearTimeout(silentTimerRef.current)
      silentTimerRef.current = setTimeout(() => {
        if (e.micOn && e.micPeak === 0) {
          const err = { name: 'SilentMicrophone', message: 'only silence for 5 s' }
          setError('Your microphone sends no sound: it may be muted (mute key or switch, or the system’s input device). Unmute it or pick another microphone, then speak again.')
          report('voice_start', err)
        }
      }, 5000)
    } catch (err) {
      setError(err.message ?? String(err))
      report('voice_start', err)
    } finally {
      setStarting(false)
    }
  }, [conv])

  // Stop voice: microphone off; typed messages are still spoken (Mute turns that off).
  const stop = useCallback(async () => {
    send({ type: 'voice', enabled: false })
    await engineRef.current?.stopMic()
    setActive(false)
    setSpeechDetected(false)
    clearTimeout(silentTimerRef.current)
  }, [conv])

  // Another microphone (Settings, or the Bluetooth hint): restart capture with it; permission is
  // already granted, so no prompt. Called from the click: the new AudioContext is made before any await
  // (AudioEngine.restartMic). A failed restart leaves the microphone off - on the server too, which
  // otherwise kept "listening" to a microphone that sent nothing.
  const restartMic = useCallback(async () => {
    const e = engineRef.current
    if (!e?.micOn) return
    try {
      await e.restartMic((pcm) => conv.connRef.current?.sendBinary(pcm))
      send({ type: 'voice', enabled: true, mic: e.micInfo })
      setMicHint(await betterThanBluetooth(e.micInfo?.device))
    } catch (err) {
      setError(err.message ?? String(err))
      report('voice_restart', err)
      send({ type: 'voice', enabled: false })
      setActive(false)
      setSpeechDetected(false)
      clearTimeout(silentTimerRef.current)
    }
  }, [conv])

  const useMic = useCallback((deviceId) => {
    setPreferredMic(deviceId)
    setMicHint(null)
    restartMic()
  }, [restartMic])

  return { active, starting, speaker, muted, level, speechDetected, error, start, stop, ensureSpeaker, toggleMute,
    engineRef, startHoldRef, micHint, useMic, restartMic, dismissMicHint: () => setMicHint(null) }
}

// per-turn diagnostics (test hooks): the recent turns only, a long session added one entry per reply
function countFor(byTurn, turnId, value) {
  byTurn[turnId] = value
  const keys = Object.keys(byTurn)
  if (keys.length > 50) delete byTurn[keys[0]]          // integer keys iterate in ascending order: the oldest
}
