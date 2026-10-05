// WebSocket connection with automatic reconnection (exponential backoff).
// Emits: 'open', 'close', 'json' (message object), 'binary' (ArrayBuffer), 'status'.

// Access token for remote use: open the page as https://host/?token=XYZ once; it is kept for the tab.
export function accessToken() {
  const fromUrl = new URLSearchParams(location.search).get('token')
  try {
    if (fromUrl) sessionStorage.setItem('avatarToken', fromUrl)
    return fromUrl ?? sessionStorage.getItem('avatarToken') ?? ''
  } catch {
    return fromUrl ?? ''
  }
}

// A random id this browser keeps: each customer has their own memory on the server, and a reconnect
// (same browser) continues its own conversation instead of waiting in line behind itself.
export function visitorId() {
  try {
    let id = localStorage.getItem('avatarVisitor')
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem('avatarVisitor', id)
    }
    return id
  } catch {
    return ''
  }
}

export function wsUrl(path = '/ws') {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const params = new URLSearchParams()
  const token = accessToken()
  if (token) params.set('token', token)
  const visitor = visitorId()
  if (visitor) params.set('visitor', visitor)
  const query = params.toString()
  return `${proto}//${location.host}${path}${query ? `?${query}` : ''}`
}

export class Connection extends EventTarget {
  constructor(url = wsUrl()) {
    super()
    this.url = url
    this.ws = null
    this.retry = 0
    this.closedByUser = false
    this.status = 'idle'
  }

  connect() {
    this.closedByUser = false
    this._setStatus('connecting')
    const ws = new WebSocket(this.url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    ws.onopen = () => {
      this.retry = 0
      this._setStatus('open')
      this.dispatchEvent(new Event('open'))
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        this.dispatchEvent(new CustomEvent('json', { detail: JSON.parse(ev.data) }))
      } else {
        this.dispatchEvent(new CustomEvent('binary', { detail: ev.data }))
      }
    }
    ws.onclose = (ev) => {
      this.dispatchEvent(new CustomEvent('close', { detail: { code: ev.code, reason: ev.reason } }))
      if (this.closedByUser || ev.code === 4001 || ev.code === 4002 || ev.code === 4003) {
        this._setStatus(ev.code === 4001 ? 'replaced' : ev.code === 4002 ? 'ended (inactive)'
          : ev.code === 4003 ? 'access denied (token)' : 'closed')
        return
      }
      const delay = Math.min(8000, 500 * 2 ** this.retry++)
      this._setStatus(`reconnecting in ${Math.round(delay / 1000)}s`)
      setTimeout(() => this.connect(), delay)
    }
  }

  sendJson(obj) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj))
  }

  sendBinary(buf) {
    if (this.ws?.readyState === WebSocket.OPEN && this.ws.bufferedAmount < 1 << 20) this.ws.send(buf)
  }

  close() {
    this.closedByUser = true
    this.ws?.close()
  }

  _setStatus(status) {
    this.status = status
    this.dispatchEvent(new CustomEvent('status', { detail: status }))
  }
}
