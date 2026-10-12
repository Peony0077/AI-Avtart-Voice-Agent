// Which microphone to use. The owner's microphone was "Headset (OpenBuds P7) (Bluetooth)": using a
// Bluetooth headset's microphone switches the headset to its phone-call profile (narrow-band audio), so
// the avatar's voice sounded thin and speech recognition got phone-quality audio. The choice is kept
// in this browser (localStorage); try/catch: storage can be unavailable (private windows).

const KEY = 'avatarMic'

// Not every "Headset": a wired or USB headset ("Headset Microphone (Realtek)", "USB Headset") has a
// full-band microphone - only the Bluetooth signs count (Windows: "(Bluetooth)", "Hands-Free AG Audio";
// Linux: bluez; AirPods and other wireless buds).
export function isBluetoothMic(label = '') {
  return /bluetooth|bluez|hands-?free|\b(hfp|hsp)\b|airpods|buds|\bbt\b/i.test(label)
}

export function preferredMic() {
  try {
    return localStorage.getItem(KEY) || ''
  } catch {
    return ''
  }
}

export function setPreferredMic(deviceId) {
  try {
    localStorage.setItem(KEY, deviceId)
  } catch { /* not kept */ }
}

// Microphones with their names (names are only shown after the page was allowed to use one).
export async function listMicrophones() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    return devices.filter((d) => d.kind === 'audioinput' && d.deviceId && d.deviceId !== 'communications')
  } catch {
    return []
  }
}

// A better choice than a Bluetooth headset's microphone, if this device has one (built-in, USB, wired).
export async function betterThanBluetooth(currentLabel) {
  if (!isBluetoothMic(currentLabel)) return null
  const mics = await listMicrophones()
  return mics.find((m) => m.deviceId !== 'default' && m.label && !isBluetoothMic(m.label)) ?? null
}
