import { useEffect, useRef, useState } from 'react'
import { useConversation } from './lib/useConversation.js'
import { useVoice } from './lib/useVoice.js'
import AvatarStage from './components/AvatarStage.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'
import SettingsDialog from './components/SettingsDialog.jsx'
import LeadsDialog from './components/LeadsDialog.jsx'

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
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [leadsOpen, setLeadsOpen] = useState(false)
  const owner = conv.role === 'owner'          // the customer link shows only the conversation
  const listRef = useRef(null)
  const busy = conv.state === 'THINKING' || conv.state === 'SPEAKING'
  // "Listening" only while the microphone is really on: the badge said Listening with the microphone
  // off, so the owner talked to her and nothing was heard
  const badge = conv.state !== 'LISTENING' ? { cls: conv.state, label: STATE_LABEL[conv.state] ?? conv.state }
    : !voice.active ? { cls: 'READY', label: 'Ready · mic off' }
      : { cls: 'LISTENING', label: voice.speechDetected ? 'Hearing you…' : 'Listening' }

  // her name and what she remembers about this person (a customer's "Forget me" button)
  useEffect(() => {
    if (conv.session) conv.loadPersona()
  }, [conv.session, conv.loadPersona])

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
    <div className={`app ${owner ? 'owner' : 'customer'}`}>
      <header className="topbar">
        <strong>{conv.session?.persona ?? 'AI Avatar'}</strong>
        <span className={`badge state-${badge.cls}`} data-testid="state" data-state={conv.state}>{badge.label}</span>
        <span className={owner ? 'muted' : 'muted conn-customer'} data-testid="conn-status">{conv.status}</span>
        <span className="voice-controls">
          {voice.active ? (
            <button type="button" className="danger" data-testid="voice-stop" onClick={voice.stop}>Stop voice</button>
          ) : (
            <button type="button" data-testid="voice-start" onClick={voice.start}
              disabled={conv.status !== 'open' || conv.session?.voice_available === false || voice.starting}>
              {voice.starting ? 'Starting…' : 'Start voice'}</button>
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
          {conv.notice && <span className="muted" data-testid="notice">{conv.notice}</span>}
        </span>
        <span className="tools">
          <button type="button" className="secondary" onClick={() => downloadTranscript(conv)}
            disabled={!conv.messages.length} data-testid="transcript">Transcript</button>
          {owner && (
            <button type="button" className="secondary" onClick={() => setLeadsOpen(true)}
              disabled={conv.status !== 'open'} data-testid="leads-open">Leads</button>
          )}
          {owner && (
            <button type="button" className="secondary" onClick={() => setSettingsOpen(true)}
              disabled={conv.status !== 'open'} data-testid="settings-open">Settings</button>
          )}
          {!owner && conv.persona?.remembered && (
            <button type="button" className="secondary" onClick={conv.forgetMemory} data-testid="forget-customer"
              title="She remembers what you told her last time; this erases it">Forget me</button>
          )}
        </span>
        {owner && conv.behavior && (
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

      {owner && <MetricsPanel metrics={conv.metrics} />}
      {owner && <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} conv={conv} voice={voice} />}
      {owner && <LeadsDialog open={leadsOpen} onClose={() => setLeadsOpen(false)} conv={conv} />}
    </div>
  )
}

// The conversation as a text file (like Anam's session transcripts), made in the page: nothing is sent.
function downloadTranscript(conv) {
  const name = conv.session?.persona ?? 'Avatar'
  const time = (at) => (at ? new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '')
  const lines = conv.messages
    .filter((m) => m.text)
    .map((m) => `[${time(m.at)}] ${m.role === 'user' ? 'You' : name}: ${m.text}${m.interrupted ? ' (interrupted)' : ''}`)
  const text = `Conversation with ${name} - ${new Date().toLocaleString()}\n\n${lines.join('\n')}\n`
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `conversation-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.txt`
  document.body.append(a)              // some browsers ignore a click on a link outside the page
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
