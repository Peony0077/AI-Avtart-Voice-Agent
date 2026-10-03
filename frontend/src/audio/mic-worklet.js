// Microphone capture worklet: Float32 input -> Int16 at 16 kHz, posted as 20 ms frames (320 samples).
// The AudioContext is created at 16 kHz where the browser allows it (Chrome, Edge). Where it runs at
// another rate (Firefox refuses a 16 kHz context for a 48 kHz microphone; some Safari versions ignore
// the requested rate) the audio is resampled here: windowed-sinc low-pass (no aliasing) + linear
// interpolation. Sending 48 kHz audio labelled as 16 kHz would make Whisper hear slow, deep speech.
const TARGET = 16000

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.frame = new Int16Array(320)
    this.fill = 0
    this.levelAcc = 0
    this.levelN = 0
    this.ratio = sampleRate / TARGET            // input samples per output sample
    if (this.ratio !== 1) {
      const taps = 63
      const cutoff = (0.45 * TARGET) / sampleRate // normalised, below the 8 kHz Nyquist of 16 kHz
      this.fir = new Float32Array(taps)
      let sum = 0
      for (let i = 0; i < taps; i++) {
        const m = i - (taps - 1) / 2
        const sinc = m === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * m) / (Math.PI * m)
        const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1))
        this.fir[i] = sinc * w
        sum += this.fir[i]
      }
      for (let i = 0; i < taps; i++) this.fir[i] /= sum
      this.hist = new Float32Array(taps)        // last filter inputs (ring)
      this.histPos = 0
      this.prev = 0                             // previous filtered sample
      this.phase = 0                            // position of the next output between prev and current
    }
  }

  emit(s) {
    s = Math.max(-1, Math.min(1, s))
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

  process(inputs) {
    const input = inputs[0]?.[0]
    if (!input) return true
    if (this.ratio === 1) {
      for (let i = 0; i < input.length; i++) this.emit(input[i])
      return true
    }
    const taps = this.fir.length
    for (let i = 0; i < input.length; i++) {
      this.hist[this.histPos] = input[i]
      this.histPos = (this.histPos + 1) % taps
      let y = 0
      for (let k = 0; k < taps; k++) y += this.fir[k] * this.hist[(this.histPos + k) % taps]
      // outputs fall between the previous filtered sample (phase 0) and this one (phase 1)
      while (this.phase <= 1) {
        this.emit(this.prev + (y - this.prev) * this.phase)
        this.phase += this.ratio
      }
      this.phase -= 1
      this.prev = y
    }
    return true
  }
}

registerProcessor('mic-capture', MicCapture)
