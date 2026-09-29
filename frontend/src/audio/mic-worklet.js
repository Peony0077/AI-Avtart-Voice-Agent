// Microphone capture worklet. Runs in an AudioContext at 16 kHz; converts
// Float32 input to Int16 and posts 20 ms frames (320 samples) to the main thread.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.frame = new Int16Array(320)
    this.fill = 0
    this.levelAcc = 0
    this.levelN = 0
  }

  process(inputs) {
    const input = inputs[0]?.[0]
    if (!input) return true
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]))
      this.frame[this.fill++] = s < 0 ? s * 0x8000 : s * 0x7fff
      this.levelAcc += s * s
      this.levelN++
      if (this.fill === this.frame.length) {
        const out = this.frame
        this.port.postMessage({ type: 'frame', pcm: out.buffer, level: Math.sqrt(this.levelAcc / this.levelN) }, [out.buffer])
        this.frame = new Int16Array(320)
        this.fill = 0
        this.levelAcc = 0
        this.levelN = 0
      }
    }
    return true
  }
}

registerProcessor('mic-capture', MicCapture)
