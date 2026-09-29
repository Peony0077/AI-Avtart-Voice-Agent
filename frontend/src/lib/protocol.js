// Binary packet header shared with backend/app/transport/protocol.py
// u8 kind | 3 reserved | u32 turn_id | u32 seq | u32 pts_ms  (little-endian, 16 bytes)
export const KIND_AUDIO = 1
export const KIND_VIDEO = 2
export const KIND_IDLE_VIDEO = 3
export const HEADER_BYTES = 16
export const TTS_SAMPLE_RATE = 24000
export const MIC_SAMPLE_RATE = 16000

export function parsePacket(buffer) {
  const view = new DataView(buffer)
  return {
    kind: view.getUint8(0),
    turnId: view.getUint32(4, true),
    seq: view.getUint32(8, true),
    ptsMs: view.getUint32(12, true),
    payload: buffer.slice(HEADER_BYTES),
  }
}
