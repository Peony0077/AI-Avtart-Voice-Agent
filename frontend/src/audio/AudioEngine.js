// Microphone capture + low-latency assistant playback (both via AudioWorklet).
import micWorkletUrl from './mic-worklet.js?url'
import playerWorkletUrl from './player-worklet.js?url'
import { MIC_SAMPLE_RATE, TTS_SAMPLE_RATE } from '../lib/protocol.js'

export class AudioEngine extends EventTarget {
  constructor() {
    super()
    this.micCtx = null
    this.playCtx = null
    this.player = null
    this.stream = null
    this.level = 0
    this.playedMs = 0
    this.playingTurn = 0
  }

  // Must be called from a user gesture (click) so the AudioContexts may start.
  async start(onMicFrame) {
    this.playCtx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE, latencyHint: 'interactive' })
    await this.playCtx.audioWorklet.addModule(playerWorkletUrl)
    this.player = new AudioWorkletNode(this.playCtx, 'pcm-player', { outputChannelCount: [1] })
    this.player.connect(this.playCtx.destination)
    this.player.port.onmessage = (e) => this._onPlayer(e.data)

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    })
    this.micCtx = new AudioContext({ sampleRate: MIC_SAMPLE_RATE, latencyHint: 'interactive' })
    await this.micCtx.audioWorklet.addModule(micWorkletUrl)
    const source = this.micCtx.createMediaStreamSource(this.stream)
    const capture = new AudioWorkletNode(this.micCtx, 'mic-capture')
    capture.port.onmessage = (e) => {
      this.level = e.data.level
      onMicFrame(e.data.pcm)
    }
    source.connect(capture)
    await Promise.all([this.playCtx.resume(), this.micCtx.resume()])
  }

  async stop() {
    this.stream?.getTracks().forEach((t) => t.stop())
    await Promise.all([this.micCtx?.close(), this.playCtx?.close()].filter(Boolean))
    this.micCtx = this.playCtx = this.player = this.stream = null
  }

  // PCM16 mono 24 kHz from the server.
  push(turnId, pcm16Buffer) {
    if (!this.player) return
    const pcm = new Int16Array(pcm16Buffer)
    const f32 = new Float32Array(pcm.length)
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768
    this.player.port.postMessage({ type: 'push', turnId, samples: f32 }, [f32.buffer])
  }

  endTurn(turnId) {
    this.player?.port.postMessage({ type: 'end', turnId })
  }

  flush() {
    this.player?.port.postMessage({ type: 'flush' })
    this.playingTurn = 0
    this.playedMs = 0
  }

  _onPlayer(msg) {
    if (msg.type === 'started') this.playingTurn = msg.turnId
    if (msg.playedMs !== undefined) this.playedMs = msg.playedMs
    this.dispatchEvent(new CustomEvent('player', { detail: msg }))
  }
}
