import { useEffect, useRef, useState } from 'react'
import { useConversation } from './lib/useConversation.js'
import { useVoice } from './lib/useVoice.js'
import AvatarStage from './components/AvatarStage.jsx'
import AvatarVoiceControls from './components/AvatarVoiceControls.jsx'
import { useMobileViewport } from './lib/useMobileViewport.js'
import MetricsPanel from './components/MetricsPanel.jsx'
import SettingsDialog from './components/SettingsDialog.jsx'
import LeadsDialog from './components/LeadsDialog.jsx'
import ConversationPanel from './components/ConversationPanel.jsx'
import Icon from './components/Icon.jsx'
import ThemeControl from './components/ThemeControl.jsx'

const SITE = 'https://h-townparty.com'
export default function App() {
  const conv = useConversation()
  const voice = useVoice(conv)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [leadsOpen, setLeadsOpen] = useState(false)
  const inputRef = useRef(null)
  const appRef = useRef(null)
  const keyboardOpen = useMobileViewport(appRef)
  const owner = conv.role === 'owner'
  const hpn = conv.session?.hpn ?? null
  const name = conv.session?.persona ?? 'our AI avatar'
  const videoSize = conv.session?.avatar?.size
  const videoAspect = videoSize?.[0] > 0 && videoSize?.[1] > 0 ? videoSize[0] / videoSize[1] : 1
  const ready = conv.status === 'open' && !!conv.session && !conv.waiting
  const status = conv.waiting ? `You're number ${conv.waiting} in line`
    : !ready ? conv.status === 'ended (inactive)' ? 'Conversation ended'
      : conv.status === 'access denied (token)' ? 'Link incomplete' : 'Connecting…'
      : conv.state === 'THINKING' || conv.state === 'END_OF_TURN' ? 'Thinking…'
        : conv.state === 'SPEAKING' ? 'Speaking'
          : voice.active ? voice.speechDetected ? 'Hearing you…' : 'Listening'
            : 'Ready to chat'
  const sessionId = conv.session?.session_id

  useEffect(() => {
    if (sessionId) conv.loadPersona()
  }, [sessionId, conv.loadPersona])
  const send = (text) => {
    if (!ready || !text.trim()) return
    voice.ensureSpeaker()
    conv.sendText(text.trim())
  }
  const typeInstead = () => {
    inputRef.current?.focus({ preventScroll: true })
  }

  return (
    <div ref={appRef} className={`app ${conv.role ?? 'pending'}${hpn ? ' hpn' : ''}${keyboardOpen ? ' keyboard-open' : ''}`}>
      <a className="skip-link" href="#conversation-input">Skip to conversation</a>
      <header className="topbar">
        <div className="topbar__inner">
          <a className="brand" href={SITE} target="_blank" rel="noopener noreferrer" aria-label="Houston Party Network website (opens in a new tab)">
            <img className="brand__logo" src="/brand/hpn-logo-white.svg" alt="Houston Party Network" width="91" height="40" />
            <span className="brand__text"><strong>HOUSTON PARTY NETWORK</strong><span>Your event. Our energy.</span></span>
          </a>
          <nav className="header-links" aria-label="HPN">
            <a href={`${SITE}/night-out/`} target="_blank" rel="noopener noreferrer">Find a night out <Icon name="external" size={14} /></a>
            <a className="header-plan" href={`${SITE}/plan-my-event/`} target="_blank" rel="noopener noreferrer">Plan my event <Icon name="arrow" size={16} /></a>
          </nav>
          <ThemeControl />
          {owner && <details className="owner-menu">
            <summary aria-label="Owner tools"><Icon name="settings" /></summary>
            <div className="owner-menu__items">
              <span>Owner tools</span>
              <button onClick={() => setSettingsOpen(true)} disabled={!ready} data-testid="settings-open">Settings</button>
              <button onClick={() => setLeadsOpen(true)} disabled={!ready} data-testid="leads-open">Leads</button>
              <MetricsPanel metrics={conv.metrics} />
            </div>
          </details>}
          <span className="visually-hidden" data-testid="conn-status">{conv.status}</span>
        </div>
      </header>

      <main className="experience">
        <h1 className="visually-hidden">Houston Party Network AI Agent</h1>

        <div className="layout" style={{'--avatar-aspect': videoAspect}}>
          <section className="stage-col" aria-label="Houston Party Network AI avatar">
            <div className="avatar-heading">
              <span className="avatar-heading__identity"><span className="avatar-heading__icon"><Icon name="agent" size={20} /></span>
                <span><strong>{conv.session?.persona ? `Meet ${name}` : 'Meet our AI avatar'}</strong><small>HPN’s AI Agent</small></span>
              </span>
              <span className={`badge ${ready ? `state-${voice.active ? conv.state : 'READY'}` : 'state-OFFLINE'}`} data-testid="state" data-state={conv.state} role="status">
                <span className="badge__dot" />{status}
              </span>
            </div>
            <div className="stage-wrap" id="agent-video">
              <AvatarStage conv={conv} voice={voice} />
              <span className="avatar-disclosure"><Icon name="sparkles" size={12} /> AI AGENT</span>
            </div>
            <AvatarVoiceControls conv={conv} voice={voice} />
            <div className="controls">
              <div className="voice-controls">
                <button type="button" className={`control-button ${voice.active ? 'is-active' : ''}`} data-testid={voice.active ? 'voice-stop' : 'voice-start'}
                  onClick={voice.active ? voice.stop : voice.start} disabled={!voice.active && (!ready || conv.session?.voice_available === false || voice.starting)}
                  aria-label={voice.active ? 'Turn microphone off' : 'Turn microphone on'} title={voice.active ? 'Turn microphone off' : 'Turn microphone on'}>
                  <Icon name={voice.active ? 'mic' : 'micOff'} size={18} /><span>{voice.active ? 'Mic on' : 'Mic off'}</span>
                </button>
                {voice.speaker && <button type="button" className="control-button" data-testid="speaker-toggle" onClick={voice.toggleMute}
                  title={voice.muted ? 'Unmute spoken replies' : 'Mute spoken replies'}>
                  <Icon name={voice.muted ? 'volumeOff' : 'volume'} size={18} />{voice.muted ? 'Unmute' : 'Mute'}
                </button>}
              </div>
              <button type="button" className="type-instead" onClick={typeInstead}><Icon name="keyboard" size={17} />Type instead</button>
            </div>
            <div className="stage-footnote">
              <span>{voice.active ? 'You can interrupt naturally. Just start speaking.' : 'Your microphone stays off until you choose to talk.'}</span>
              {voice.active && <span className={`mic ${voice.speechDetected ? 'hot' : ''}`} data-testid="mic"><span className="mic-bar" style={{width: `${Math.min(100, voice.level * 400)}%`}} /></span>}
            </div>
            {conv.notice && <p className="notice" role="status" data-testid="notice">{conv.notice}</p>}
          </section>

          <ConversationPanel conv={conv} ready={ready} onSend={send} inputRef={inputRef} />
        </div>


      </main>

      <footer className="site-footer">
        <div className="site-footer__inner"><span className="footer-location"><Icon name="pin" size={15} />Made for good times in Greater Houston.</span>
          <nav aria-label="Help and information">
            <a href={`${SITE}/contact-us/`} target="_blank" rel="noopener noreferrer">Talk to our team <Icon name="external" size={12} /></a>
            <a href={`${SITE}/terms-and-privacy/`} target="_blank" rel="noopener noreferrer">Privacy</a>
            <a href={SITE} target="_blank" rel="noopener noreferrer">h-townparty.com <Icon name="external" size={12} /></a>
          </nav>
        </div>
      </footer>
      {owner && <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} conv={conv} voice={voice} />}
      {owner && <LeadsDialog open={leadsOpen} onClose={() => setLeadsOpen(false)} conv={conv} />}
    </div>
  )
}
