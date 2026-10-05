// Binary packet header shared with backend/app/transport/protocol.py
// u8 kind | u8 flags | 2 reserved | u32 turn_id | u32 seq | u32 pts_ms  (little-endian, 16 bytes)
// flags (avatar speech frames): FLAG_TAIL = the reply's closing tail (after its last word)
export const KIND_AUDIO = 1
export const KIND_VIDEO = 2
export const KIND_IDLE_VIDEO = 3
export const KIND_VIDEO_H264 = 4     // the same frames as H.264 access units (lib/h264.js)
export const KIND_IDLE_H264 = 5
export const KIND_AUDIO_OPUS = 6       // the voice as Ogg Opus: u32 sample count, then the file (lib/opus.js)
export const HEADER_BYTES = 16
export const FLAG_TAIL = 1
export const FLAG_KEY = 2              // H.264 keyframe
export const TTS_SAMPLE_RATE = 24000
export const MIC_SAMPLE_RATE = 16000

export function parsePacket(buffer) {
  const view = new DataView(buffer)
  return {
    kind: view.getUint8(0),
    flags: view.getUint8(1),
    turnId: view.getUint32(4, true),
    seq: view.getUint32(8, true),
    ptsMs: view.getUint32(12, true),
    payload: buffer.slice(HEADER_BYTES),
  }
}
