import { useEffect, useRef, useState } from 'react'
import { useConversation } from './lib/useConversation.js'
import { useVoice } from './lib/useVoice.js'
import AvatarStage from './components/AvatarStage.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'

const STATE_LABEL = {
  IDLE: 'Idle',
  LISTENING: 'Listening',
  END_OF_TURN: 'Got it…',
  THINKING: 'Thinking…',
  SPEAKING: 'Speaking',
  INTERRUPTED: 'Interrupted',
  ERROR: 'Error',
}

export default function App() {
  const conv = useConversation()
  const voice = useVoice(conv)
  const [draft, setDraft] = useState('')
  const listRef = useRef(null)
  const busy = conv.state === 'THINKING' || conv.state === 'SPEAKING'

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight })
  }, [conv.messages])

  const submit = (e) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text) return
    voice.ensureSpeaker()        // typed messages are spoken (and lip-synced) too; needs this user gesture
    conv.sendText(text)
    setDraft('')
  }

  return (
    <div className="app">
      <header className="topbar">
        <strong>{conv.session?.persona ?? 'AI Avatar'}</strong>
        <span className={`badge state-${conv.state}`} data-testid="state">{STATE_LABEL[conv.state] ?? conv.state}</span>
        <span className="muted" data-testid="conn-status">{conv.status}</span>
        <span className="voice-controls">
          {voice.active ? (
            <button type="button" className="danger" data-testid="voice-stop" onClick={voice.stop}>Stop voice</button>
          ) : (
            <button type="button" data-testid="voice-start" onClick={voice.start}
              disabled={conv.status !== 'open' || conv.session?.voice_available === false}>Start voice</button>
          )}
          {voice.active && (
            <span className={`mic ${voice.speechDetected ? 'hot' : ''}`} data-testid="mic" title="microphone level">
              <span className="mic-bar" style={{ width: `${Math.min(100, voice.level * 400)}%` }} />
            </span>
          )}
          {voice.speaker && (
            <button type="button" className="secondary" data-testid="speaker-toggle" onClick={voice.toggleMute}
              title={voice.muted ? 'Replies are text only' : 'Replies are spoken'}>
              {voice.muted ? 'Unmute' : 'Mute'}
            </button>
          )}
          {voice.error && <span className="error" data-testid="voice-error">{voice.error}</span>}
        </span>
        {conv.behavior && (
          <span className="muted" data-testid="behavior">
            {conv.behavior.emotion} · {conv.behavior.gesture} · {conv.behavior.intensity}
          </span>
        )}
      </header>

      <main className="layout">
        <AvatarStage conv={conv} voice={voice} />

        <section className="chat" aria-label="Conversation">
          <ol className="messages" ref={listRef} data-testid="messages">
            {conv.messages.map((m, i) => (
              <li key={i} className={`msg ${m.role}`} data-role={m.role} data-turn={m.turnId}>
                <span className="text">{m.text || (m.role === 'assistant' ? '…' : '')}</span>
                {m.interrupted && <span className="tag">interrupted</span>}
                {m.citations?.length > 0 && (
                  <ul className="sources" data-testid="sources">
                    {m.citations.map((c) => (
                      <li key={c.n} title={c.snippet}>
                        [{c.n}] {c.title}{c.section ? ` · ${c.section}` : ''}{c.page ? ` · p.${c.page}` : ''} <span className="muted">({c.score})</span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ol>
          <form className="composer" onSubmit={submit}>
            <input
              aria-label="Message"
              data-testid="input"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type a message…"
              autoComplete="off"
            />
            <button type="submit" data-testid="send" disabled={!draft.trim() || conv.status !== 'open'}>Send</button>
            <button type="button" data-testid="stop" onClick={conv.interrupt} disabled={!busy} className="secondary">
              Stop
            </button>
          </form>
        </section>
      </main>

      <MetricsPanel metrics={conv.metrics} />
    </div>
  )
}
