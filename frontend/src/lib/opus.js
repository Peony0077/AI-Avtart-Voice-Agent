// The voice as Ogg Opus (backend app/tts/opus.py): ~56 kbit/s instead of 384 for PCM, so a slow connection
// still carries the voice and the video (owner, 2026-10-05: "make the avatar light and smooth"). Decoded
// here and cut to the exact sample count the server sends along: the lip-sync clock (samples played)
// stays exact. ?pcm: PCM as before (tests that analyse the samples).
import { TTS_SAMPLE_RATE } from './protocol.js'

// 50 ms of silence as Ogg Opus (ffmpeg -f lavfi -i anullsrc=r=48000:cl=mono -t 0.05 -c:a libopus -f ogg).
const PROBE = 'T2dnUwACAAAAAAAAAAAAAAAAAAAAAAIotXIBE09wdXNIZWFkAQE4AYC7AAAAAABPZ2dTAAAAAAAAAAAAAAAAAAABAAAASZW+VAEu'
  + 'T3B1c1RhZ3MGAAAAZmZtcGVnAQAAABQAAABlbmNvZGVyPUxhdmMgbGlib3B1c09nZ1MABJgKAAAAAAAAAAAAAAIAAABaXtsSAw8REngL'
  + '5ME27MWNjElGmaQmsHgHyXIn4UTqVfHwwNXd6VJgeAfJecjJV8CiEiP672fzZMDg'

let decoder = null

// Whether this browser really decodes Ogg Opus (true/false, once): an OfflineAudioContext alone said
// nothing - where decodeAudioData refused the file, every reply played as silence. The page then asks
// for PCM (the "hello", lib/useConversation.js).
export const opusSupport = (async () => {
  try {
    if (typeof OfflineAudioContext === 'undefined' || new URLSearchParams(location.search).has('pcm')) return false
    const bytes = Uint8Array.from(atob(PROBE), (c) => c.charCodeAt(0))
    decoder ??= new OfflineAudioContext(1, 1, TTS_SAMPLE_RATE)
    const audio = await Promise.race([decoder.decodeAudioData(bytes.buffer),
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 2000))])
    return audio.length > 0
  } catch {
    return false
  }
})()
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
