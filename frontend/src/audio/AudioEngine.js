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
    this.turnStartCtx = new Map()
    this.endedTurns = new Set()
  }

  // Must be called from a user gesture (click) so the AudioContexts may start.
  async start(onMicFrame) {
    await this.startPlayback()
    await this.startMic(onMicFrame)
  }

  // Assistant audio only (typed messages are spoken too). Call from a user gesture: the
  // AudioContext is created synchronously, before the first await, so the browser lets it run.
  async startPlayback() {
    if (this.playCtx) return
    this.playCtx = new AudioContext({ sampleRate: TTS_SAMPLE_RATE, latencyHint: 'interactive' })
    const ready = (async () => {
      await this.playCtx.audioWorklet.addModule(playerWorkletUrl)
      this.player = new AudioWorkletNode(this.playCtx, 'pcm-player', { outputChannelCount: [1] })
      this.player.connect(this.playCtx.destination)
      this.player.port.onmessage = (e) => this._onPlayer(e.data)
      await this.playCtx.resume()
    })()
    this._playbackReady = ready
    await ready
  }

  get micOn() {
    return Boolean(this.stream)
  }

  async startMic(onMicFrame) {
    if (this.stream) return
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
    await this.micCtx.resume()
  }

  async stopMic() {
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.micCtx?.close()
    this.micCtx = this.stream = null
    this.level = 0
  }

  async stop() {
    await this.stopMic()
    await this.playCtx?.close()
    this.playCtx = this.player = null
  }

  // PCM16 mono 24 kHz from the server.
  push(turnId, pcm16Buffer) {
    if (!this.player) {
      // playback still starting (first typed message): queue until the worklet exists
      this._playbackReady?.then(() => this.push(turnId, pcm16Buffer))
      return
    }
    const pcm = new Int16Array(pcm16Buffer)
    const f32 = new Float32Array(pcm.length)
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768
    this.player.port.postMessage({ type: 'push', turnId, samples: f32 }, [f32.buffer])
  }

  endTurn(turnId) {
    if (!this.player) {
      this._playbackReady?.then(() => this.endTurn(turnId))
      return
    }
    this.player.port.postMessage({ type: 'end', turnId })
  }

  flush() {
    this.player?.port.postMessage({ type: 'flush' })
    if (this.playingTurn) this.endedTurns.add(this.playingTurn)
    this.playingTurn = 0
    this.playedMs = 0
  }

  // Position (ms) on a turn's audio timeline that is audible right now, or null.
  audibleMs(turnId) {
    const start = this.turnStartCtx.get(turnId)
    if (start === undefined || !this.playCtx) return null
    const ts = this.playCtx.getOutputTimestamp?.()
    const nowCtx = ts && ts.contextTime > 0
      ? ts.contextTime + (performance.now() - ts.performanceTime) / 1000
      : this.playCtx.currentTime - (this.playCtx.outputLatency || 0)
    return (nowCtx - start) * 1000
  }

  _onPlayer(msg) {
    if (msg.type === 'started') {
      this.playingTurn = msg.turnId
      this.turnStartCtx.set(msg.turnId, msg.ctxTime)
    }
    if (msg.type === 'ended' || msg.type === 'flushed') this.endedTurns.add(msg.turnId ?? this.playingTurn)
    if (msg.playedMs !== undefined) this.playedMs = msg.playedMs
    this.dispatchEvent(new CustomEvent('player', { detail: msg }))
  }
}
