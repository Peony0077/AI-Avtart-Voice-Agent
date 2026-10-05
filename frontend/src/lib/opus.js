// The voice as Ogg Opus (backend app/tts/opus.py): ~56 kbit/s instead of 384 for PCM, so a slow connection
// still carries the voice and the video (owner, 2026-10-05: "make the avatar light and smooth"). Decoded
// here and cut to the exact sample count the server sends along: the lip-sync clock (samples played)
// stays exact. ?pcm: PCM as before (tests that analyse the samples).
import { TTS_SAMPLE_RATE } from './protocol.js'

export const opusSupport = typeof OfflineAudioContext !== 'undefined'
  && !new URLSearchParams(location.search).has('pcm')

let decoder = null
export const opusStats = { decoded: 0, failed: 0 }

// payload: ArrayBuffer (u32 sample count, Ogg Opus) -> Float32Array at 24 kHz, exactly that long
export async function decodeOpus(payload) {
  const samples = new DataView(payload).getUint32(0, true)
  const out = new Float32Array(samples)
  try {
    decoder ??= new OfflineAudioContext(1, 1, TTS_SAMPLE_RATE)
    const audio = await decoder.decodeAudioData(payload.slice(4))
    out.set(audio.getChannelData(0).subarray(0, samples))
    opusStats.decoded++
  } catch {
    opusStats.failed++         // silence of the right length: the voice and the lips stay in step
  }
  return out
}
