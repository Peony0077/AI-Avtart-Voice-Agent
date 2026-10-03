// Streamed PCM player worklet (AudioContext at 24 kHz).
// Ring buffer of queued samples tagged by turn. Messages in:
//   {type:'push', turnId, samples: Float32Array}   append audio of a turn
//   {type:'end', turnId}                           no more audio will come for this turn
//   {type:'flush'}                                 drop everything immediately (barge-in)
// Messages out:
//   {type:'started', turnId}                       first sample of a turn reached the output
//   {type:'progress', turnId, playedMs, gapMs}     every ~100 ms while playing
//   {type:'ended', turnId, playedMs, gapMs}        all audio of an ended turn has been played
// gapMs: time the player ran dry in the middle of a turn (started, more audio still to come) =
// audible gaps in the reply.
//   {type:'flushed'}
const CAPACITY = 24000 * 180 // 3 minutes

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
    this.endedTurns = new Set()
    this.startedTurns = new Set()
    this.sinceReport = 0
    this.port.onmessage = (e) => this.onMessage(e.data)
  }

  onMessage(msg) {
    if (msg.type === 'push') {
      if (msg.turnId !== this.turnId) {
        // A new turn starts: anything still queued belongs to an older turn → drop it.
        this.read = this.write = this.size = 0
        this.turnId = msg.turnId
        this.played = 0
        this.gap = 0
      }
      const s = msg.samples
      const n = Math.min(s.length, CAPACITY - this.size)
      for (let i = 0; i < n; i++) {
        this.buf[this.write] = s[i]
        this.write = (this.write + 1) % CAPACITY
      }
      this.size += n
    } else if (msg.type === 'end') {
      this.endedTurns.add(msg.turnId)
    } else if (msg.type === 'flush') {
      this.read = this.write = this.size = 0
      this.turnId = 0
      this.played = 0
      this.gap = 0
      this.port.postMessage({ type: 'flushed' })
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
    if (take < n && this.turnId && this.startedTurns.has(this.turnId) && !this.endedTurns.has(this.turnId)) {
      this.gap += n - take                     // ran dry mid-reply: the listener hears a gap
    }

    if (take > 0) {
      if (!this.startedTurns.has(this.turnId)) {
        this.startedTurns.add(this.turnId)
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
    if (this.size === 0 && this.turnId && this.endedTurns.has(this.turnId)) {
      this.endedTurns.delete(this.turnId)
      this.port.postMessage({ type: 'ended', turnId: this.turnId, playedMs: (this.played / sampleRate) * 1000, gapMs: (this.gap / sampleRate) * 1000 })
    }
    return true
  }
}

registerProcessor('pcm-player', PcmPlayer)
