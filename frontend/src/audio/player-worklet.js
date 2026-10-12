// Streamed PCM player worklet (AudioContext at 24 kHz).
// Ring buffer of queued samples tagged by turn. Messages in:
//   {type:'push', turnId, samples: Float32Array}   append audio of a turn
//   {type:'end', turnId}                           no more audio will come for this turn
//   {type:'flush'}                                 drop everything immediately (barge-in)
//   {type:'reset'}                                 flush and forget every turn id (a new session)
// Messages out:
//   {type:'started', turnId, ctxTime}              first sample of a turn reached the output
//   {type:'progress', turnId, playedMs, gapMs}     every ~100 ms while playing
//   {type:'dry', turnId, ctxTime}                  ran dry mid-reply from ctxTime on (a gap starts)
//   {type:'resumed', turnId, ctxTime}              the turn's audio continues at ctxTime (the gap ends)
//   {type:'ended', turnId, playedMs, gapMs}        all audio of an ended turn has been played
// gapMs: time the player ran dry in the middle of a turn (started, more audio still to come) =
// audible gaps in the reply. dry/resumed let the page's lip-sync clock stop during a gap: frame pts
// are positions on the audio timeline, which a gap does not advance (AudioEngine.audibleMs).
//   {type:'flushed'}
const CAPACITY = 24000 * 180 // 3 minutes
const KEEP_TURNS = 16        // turn ids remembered (started / ended): only recent ones matter

class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super()
    this.buf = new Float32Array(CAPACITY)
    this.read = 0
    this.write = 0
    this.size = 0
    this.turnId = 0          // turn whose audio is at the read head
    this.played = 0          // samples played for turnId
    this.gap = 0             // samples of silence while waiting for more audio of turnId
    this.dry = false         // in such a gap now
    this.done = false        // 'ended' was posted for turnId: the silence after it is no gap
    this.endedTurns = new Set()
    this.startedTurns = new Set()
    this.sinceReport = 0
    this.port.onmessage = (e) => this.onMessage(e.data)
  }

  newTurn(turnId) {
    this.read = this.write = this.size = 0
    this.turnId = turnId
    this.played = 0
    this.gap = 0
    this.dry = false
    this.done = false
  }

  onMessage(msg) {
    if (msg.type === 'push') {
      // A new turn starts: anything still queued belongs to an older turn → drop it.
      if (msg.turnId !== this.turnId) this.newTurn(msg.turnId)
      const s = msg.samples
      const n = Math.min(s.length, CAPACITY - this.size)
      for (let i = 0; i < n; i++) {
        this.buf[this.write] = s[i]
        this.write = (this.write + 1) % CAPACITY
      }
      this.size += n
    } else if (msg.type === 'end') {
      remember(this.endedTurns, msg.turnId)
    } else if (msg.type === 'flush' || msg.type === 'reset') {
      this.newTurn(0)
      if (msg.type === 'reset') {        // a new session: its turn ids start again at 1
        this.endedTurns.clear()
        this.startedTurns.clear()
      } else {
        this.port.postMessage({ type: 'flushed' })
      }
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0][0]
    const n = out.length
    const take = Math.min(n, this.size)
    for (let i = 0; i < take; i++) {
      out[i] = this.buf[this.read]
      this.read = (this.read + 1) % CAPACITY
    }
    for (let i = take; i < n; i++) out[i] = 0
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out)
    this.size -= take
    const midTurn = this.turnId && !this.done && this.startedTurns.has(this.turnId) && !this.endedTurns.has(this.turnId)
    if (this.dry && (take > 0 || !midTurn)) {
      // audio again (from this quantum's first sample), or the turn ended while dry: the clock runs on
      this.dry = false
      this.port.postMessage({ type: 'resumed', turnId: this.turnId, ctxTime: currentTime })
    }
    if (take < n && midTurn) {
      this.gap += n - take                     // ran dry mid-reply: the listener hears a gap
      if (!this.dry) {
        this.dry = true
        this.port.postMessage({ type: 'dry', turnId: this.turnId, ctxTime: currentTime + take / sampleRate })
      }
    }

    if (take > 0) {
      if (!this.startedTurns.has(this.turnId)) {
        remember(this.startedTurns, this.turnId)
        // currentTime = context time of this render quantum's first sample = when this turn's audio starts
        this.port.postMessage({ type: 'started', turnId: this.turnId, ctxTime: currentTime })
      }
      this.played += take
      this.sinceReport += take
      if (this.sinceReport >= 2400) {
        this.sinceReport = 0
        this.port.postMessage({ type: 'progress', turnId: this.turnId, playedMs: (this.played / sampleRate) * 1000, gapMs: (this.gap / sampleRate) * 1000 })
      }
    }
    if (this.size === 0 && this.turnId && !this.done && this.endedTurns.has(this.turnId)) {
      this.endedTurns.delete(this.turnId)
      this.done = true
      this.port.postMessage({ type: 'ended', turnId: this.turnId, playedMs: (this.played / sampleRate) * 1000, gapMs: (this.gap / sampleRate) * 1000 })
    }
    return true
  }
}

// add to a set of turn ids, forgetting the oldest beyond KEEP_TURNS (a long session added one per reply)
function remember(set, id) {
  set.add(id)
  if (set.size > KEEP_TURNS) set.delete(set.values().next().value)
}

registerProcessor('pcm-player', PcmPlayer)
