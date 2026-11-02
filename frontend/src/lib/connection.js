// WebSocket connection with automatic reconnection (exponential backoff).
// Emits: 'open', 'close', 'json' (message object), 'binary' (ArrayBuffer), 'status'.

// Access token for remote use: open the page as https://host/?token=XYZ once; it is kept for the tab.
export function accessToken() {
  const fromUrl = new URLSearchParams(location.search).get('token')
  try {
    if (fromUrl) {
      sessionStorage.setItem('avatarToken', fromUrl)
      // kept for the tab: out of the address bar, so it is not in the history, a screenshot or a link
      // copied from the bar (the owner's token gives settings and leads)
      const url = new URL(location.href)
      url.searchParams.delete('token')
      history.replaceState(history.state, '', url.pathname + url.search + url.hash)
    }
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
      id = randomId()
      localStorage.setItem('avatarVisitor', id)
    }
    return id
  } catch {
    return ''
  }
}

// crypto.randomUUID exists only on a secure page (https, localhost): on plain http (a LAN address) the
// id was '' - no memory of the customer, and a reconnect waited in line behind itself
function randomId() {
  if (crypto.randomUUID) return crypto.randomUUID()
  const b = crypto.getRandomValues(new Uint8Array(16))
  b[6] = (b[6] & 0x0f) | 0x40                      // the shape of a version-4 UUID
  b[8] = (b[8] & 0x3f) | 0x80
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('')
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
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
    this.timer = 0
  }

  connect() {
    clearTimeout(this.timer)
    this.closedByUser = false
    this._setStatus('connecting')
    const ws = new WebSocket(this.url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    this.received = this.acked = this.ackedAt = 0       // a new connection: the server counts from 0
    ws.onopen = () => {
      this.retry = 0
      this._setStatus('open')
      this.dispatchEvent(new Event('open'))
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        this.dispatchEvent(new CustomEvent('json', { detail: JSON.parse(ev.data) }))
      } else {
        this._ack(ev.data.byteLength)
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
      // 1013: the line of waiting customers is full (api/ws.py MAX_WAITING) - try again a little later;
      // 1012: the server is restarting
      const delay = ev.code === 1013 ? 15000 : Math.min(8000, 500 * 2 ** this.retry++)
      const why = ev.code === 1013 ? 'many people are waiting - trying again' : ev.code === 1012 ? 'server restarting - reconnecting'
        : 'reconnecting'
      this._setStatus(`${why} in ${Math.round(delay / 1000)}s`)
      this.timer = setTimeout(() => this.connect(), delay)
    }
  }

  sendJson(obj) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj))
  }

  sendBinary(buf) {
    if (this.ws?.readyState === WebSocket.OPEN && this.ws.bufferedAmount < 1 << 20) this.ws.send(buf)
  }

  // The server sends video only while little of it is unacknowledged (backend outbox.py), so on a slow
  // connection the voice never waits behind seconds of video on its way: tell it what has arrived.
  _ack(bytes) {
    this.received += bytes
    const now = performance.now()
    if (this.received - this.acked >= 16384 || now - this.ackedAt >= 100) {
      this.acked = this.received
      this.ackedAt = now
      this.sendJson({ type: 'ack', bytes: this.received })
    }
  }

  // For good: no reconnect - also not one already waiting for its backoff (a Connection closed then
  // reconnected anyway) - and no more events from its socket.
  close() {
    this.closedByUser = true
    clearTimeout(this.timer)
    const ws = this.ws
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = null
      ws.close()
    }
    this._setStatus('closed')
  }

  _setStatus(status) {
    this.status = status
    this.dispatchEvent(new CustomEvent('status', { detail: status }))
  }
}
