// H.264 video from the avatar worker (avatar/musetalk/h264.py), decoded with WebCodecs.
// Owner, 2026-10-05: "delay and non-smooth" through the tunnel - one JPEG per frame needed 6.7 Mbit/s,
// H.264 ~1 Mbit/s for the same picture. Decoded frames become ImageBitmaps (the VideoFrame is closed at
// once: a hardware decoder stalls when its frames are held), so the avatar view keeps drawing them by
// the audio clock exactly as before. WebCodecs needs a secure page (https or localhost); elsewhere, or
// with ?jpeg, the page asks for JPEG frames.
// High profile, level 4.1 (avc1.640029): the worker encodes frames up to 1024x1024 - 4096 macroblocks,
// more than Baseline 3.1's 3600 (avc1.42E01F) allows. The probe asks for that size, so a decoder that
// cannot take it means JPEG frames instead of a broken picture.

const CONFIG = { codec: 'avc1.640029', optimizeForLatency: true }

export const h264Support = (async () => {
  try {
    if (new URLSearchParams(location.search).has('jpeg')) return false
    if (typeof VideoDecoder === 'undefined') return false
    return !!(await VideoDecoder.isConfigSupported({ ...CONFIG, codedWidth: 1024, codedHeight: 1024 })).supported
  } catch {
    return false
  }
})()

// One stream (the idle video, or one reply): frames in order; onFrame(meta, bitmap) for each decoded one.
// fps: the avatar's frame rate (20 or 25, the session's avatar.fps) - each chunk's timestamp advances by
// one real frame duration.
export class H264Stream {
  constructor(onFrame, onError, fps = 20) {
    this.meta = []                 // what each queued chunk is (idx, pts, ...), in decode order
    this.needKey = true            // a decoder starts at a keyframe
    this.broken = false
    this.ts = 0
    this.frameUs = Math.round(1e6 / (fps > 0 ? fps : 20))
    this.decoder = new VideoDecoder({
      output: (frame) => {
        const meta = this.meta.shift()
        createImageBitmap(frame).then(
          (bmp) => { frame.close(); onFrame(meta, bmp) },
          () => frame.close(),
        )
      },
      error: (e) => { this.broken = true; onError?.(e) },
    })
    this.decoder.configure(CONFIG)
  }

  // false: not decoded (waiting for a keyframe, or the decoder failed)
  feed(payload, key, meta) {
    if (this.broken) return false
    if (this.needKey) {
      if (!key) return false
      this.needKey = false
    }
    this.meta.push(meta)
    try {
      this.decoder.decode(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp: this.ts++ * this.frameUs,
        data: new Uint8Array(payload) }))
      return true
    } catch {
      this.broken = true
      return false
    }
  }

  close() {
    this.meta = []
    try { this.decoder.close() } catch { /* already closed */ }
  }
}
