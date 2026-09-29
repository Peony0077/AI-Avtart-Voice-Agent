import { useEffect, useRef, useState } from 'react'
import { KIND_IDLE_VIDEO, KIND_VIDEO, parsePacket } from '../lib/protocol.js'

// Renders the avatar video. Idle frames: newest wins. Speech frames: shown by the
// audio clock — the frame whose pts (ms on the turn's audio timeline) matches what is
// audible right now, so lips stay in sync regardless of network jitter.
export default function AvatarStage({ conv, voice }) {
  const canvasRef = useRef(null)
  const idleRef = useRef(null)               // ImageBitmap
  const speechRef = useRef(new Map())        // turnId -> [{pts, bmp}] sorted by pts
  const rejectUpToRef = useRef(0)
  const statsRef = useRef({ idle: 0, speech: 0, drawnSpeech: 0, late: 0, avOffsets: [], arrivals: [], clock: null })
  const [size, setSize] = useState(null)

  useEffect(() => {
    const offJson = conv.onJson((msg) => {
      if (msg.type === 'session' && msg.avatar?.size) setSize(msg.avatar.size)
      if (msg.type === 'interrupt') {
        rejectUpToRef.current = Math.max(rejectUpToRef.current, msg.turn_id)
        dropTurnsUpTo(speechRef.current, msg.turn_id)
      }
    })
    const offBinary = conv.onBinary(async (buffer) => {
      const p = parsePacket(buffer)
      if (p.kind !== KIND_IDLE_VIDEO && p.kind !== KIND_VIDEO) return
      if (p.kind === KIND_VIDEO && p.turnId <= rejectUpToRef.current) return
      const bmp = await createImageBitmap(new Blob([p.payload], { type: 'image/jpeg' }))
      if (p.kind === KIND_IDLE_VIDEO) {
        statsRef.current.idle++
        idleRef.current?.close()
        idleRef.current = bmp
      } else {
        statsRef.current.speech++
        const eng = voice.engineRef.current
        const audibleNow = eng?.playingTurn === p.turnId ? eng.audibleMs(p.turnId) : null
        const st = statsRef.current
        if (st.arrivals.length < 600) st.arrivals.push(audibleNow === null ? null : Math.round(audibleNow - p.ptsMs))
        const list = speechRef.current.get(p.turnId) ?? []
        list.push({ pts: p.ptsMs, bmp })
        list.sort((a, b) => a.pts - b.pts)
        speechRef.current.set(p.turnId, list)
      }
    })
    window.__avatarStats = statsRef.current // test hook
    return () => { offJson(); offBinary() }
  }, [conv, voice])

  useEffect(() => {
    let raf = 0
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      let frame = null
      const engine = voice.engineRef.current
      const turn = engine?.playingTurn
      const list = turn ? speechRef.current.get(turn) : null
      if (engine && turn && list?.length && !engine.endedTurns.has(turn)) {
        const audible = engine.audibleMs(turn)
        if (!statsRef.current.clock && engine.playCtx) {
          const ts = engine.playCtx.getOutputTimestamp?.()
          statsRef.current.clock = { outputLatency: engine.playCtx.outputLatency, baseLatency: engine.playCtx.baseLatency,
            currentTime: engine.playCtx.currentTime, tsContext: ts?.contextTime, audible, start: engine.turnStartCtx.get(turn) }
        }
        if (audible !== null && audible >= 0) {
          // newest frame not in the future; free the ones we have passed
          let i = -1
          while (i + 1 < list.length && list[i + 1].pts <= audible) i++
          if (i >= 0) {
            frame = list[i].bmp
            const st = statsRef.current
            st.drawnSpeech++
            st.avOffsets.push(audible - list[i].pts)
            if (st.avOffsets.length > 400) st.avOffsets.shift()
            if (audible - list[i].pts > 120) st.late++
            for (const old of list.splice(0, i)) old.bmp.close()
          }
        }
      }
      if (!frame) frame = idleRef.current
      if (frame) {
        if (canvas.width !== frame.width) { canvas.width = frame.width; canvas.height = frame.height }
        ctx.drawImage(frame, 0, 0)
      }
      // forget finished turns
      for (const [id, l] of speechRef.current) {
        if (engine?.endedTurns.has(id)) { l.forEach((f) => f.bmp.close()); speechRef.current.delete(id) }
      }
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [voice])

  const available = conv.session?.avatar_available
  return (
    <section className="stage" aria-label="Avatar">
      {available ? (
        <canvas ref={canvasRef} data-testid="avatar-canvas" width={size?.[0] ?? 512} height={size?.[1] ?? 512} />
      ) : (
        <div className="stage-placeholder">Avatar disabled (start with profile “avatar”)</div>
      )}
    </section>
  )
}

function dropTurnsUpTo(map, turnId) {
  for (const [id, list] of map) {
    if (id <= turnId) { list.forEach((f) => f.bmp.close()); map.delete(id) }
  }
}
