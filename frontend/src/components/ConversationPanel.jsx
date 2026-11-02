import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import HelperReply, { HelperWelcome } from './HelperReply.jsx'
import Icon from './Icon.jsx'

const TOPICS = [
  { icon: 'calendar', label: 'Plan an event', question: 'Help me plan an event' },
  { icon: 'heart', label: 'Weddings', question: 'Tell me about weddings' },
  { icon: 'music', label: 'Nights out', question: "What's on tonight?" },
  { icon: 'ticket', label: 'Packages & prices', question: 'Prices and packages' },
]
const TIME = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' })

function replyText(message) {
  if (!message.reply) return message.text || ''
  return [message.reply.text, ...(message.reply.cards || []).map(card =>
    [card.title, card.subtitle, card.detail].filter(Boolean).join(' — '))].filter(Boolean).join('\n\n')
}

export default function ConversationPanel({ conv, ready, onSend, inputRef }) {
  const [draft, setDraft] = useState('')
  const [showLatest, setShowLatest] = useState(false)
  const [copied, setCopied] = useState(null)
  const [copyNotice, setCopyNotice] = useState('')
  const [topicsOpen, setTopicsOpen] = useState(false)
  const topicsRef = useRef(null)
  const topicsButtonRef = useRef(null)
  const listRef = useRef(null)
  const followLatest = useRef(true)
  const lastAligned = useRef(-1)     // the last answer whose start was brought into view
  const copyTimer = useRef(null)
  const hpn = conv.session?.hpn
  const name = conv.session?.persona || 'AI Agent'
  const busy = ['THINKING', 'END_OF_TURN', 'SPEAKING'].includes(conv.state)
  const lastAssistant = conv.messages.findLastIndex(message => message.role === 'assistant')
  const last = conv.messages.at(-1)
  const awaitingReply = busy && (!last || last.role === 'user')
  // the topic bar before the first message; after it, behind the Topics button (more room for the answers)
  const topicsInline = hpn && conv.messages.length === 0
  const subtitle = conv.waiting ? `You’re number ${conv.waiting} in line`
    : !ready ? conv.status === 'ended (inactive)' ? 'Conversation ended — select Talk again to reconnect' : `Connecting to ${name}…`
      : conv.state === 'SPEAKING' ? `${name} is speaking`
        : busy ? `${name} is preparing your answer` : 'Here to help with your next good time'

  useLayoutEffect(() => {
    const list = listRef.current
    if (!list) return
    // A new formatted answer (prices, tonight's nights) is shown from its start: following the bottom
    // opened a long answer at its buttons, and the price list had to be scrolled back to.
    const index = conv.messages.length - 1
    const newest = conv.messages[index]
    if (newest?.role === 'assistant' && newest.reply && index !== lastAligned.current) {
      lastAligned.current = index
      const item = list.querySelector(`li[data-index="${index}"]`)
      if (item && item.offsetHeight > list.clientHeight * 0.7) {
        list.scrollTop += item.getBoundingClientRect().top - list.getBoundingClientRect().top - 12
        followLatest.current = false
        setShowLatest(true)
        return
      }
    }
    if (followLatest.current) list.scrollTop = list.scrollHeight
  }, [conv.messages, awaitingReply])

  useLayoutEffect(() => {
    const field = inputRef.current
    if (!field) return
    field.style.height = 'auto'
    field.style.height = `${Math.min(field.scrollHeight, 108)}px`
  }, [draft, inputRef])

  useEffect(() => () => clearTimeout(copyTimer.current), [])

  useEffect(() => {
    if (!topicsOpen) return
    const dismiss = event => {
      if (!topicsRef.current?.contains(event.target) && !topicsButtonRef.current?.contains(event.target)) setTopicsOpen(false)
    }
    const escape = event => {
      if (event.key !== 'Escape') return
      setTopicsOpen(false)
      topicsButtonRef.current?.focus({ preventScroll: true })
    }
    document.addEventListener('pointerdown', dismiss)
    document.addEventListener('focusin', dismiss)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', dismiss)
      document.removeEventListener('focusin', dismiss)
      document.removeEventListener('keydown', escape)
    }
  }, [topicsOpen])

  useEffect(() => {
    const list = listRef.current
    if (!list) return
    const observer = new ResizeObserver(() => {
      if (followLatest.current) list.scrollTop = list.scrollHeight
    })
    observer.observe(list)
    return () => observer.disconnect()
  }, [])

  const scrollToLatest = () => {
    followLatest.current = true
    setShowLatest(false)
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }
  const trackScroll = event => {
    const list = event.currentTarget
    const nearBottom = list.scrollHeight - list.clientHeight - list.scrollTop < 48
    followLatest.current = nearBottom
    setShowLatest(!nearBottom)
  }
  const send = text => {
    if (!ready || !text.trim()) return
    followLatest.current = true
    setShowLatest(false)
    onSend(text.trim())
  }
  const submit = event => {
    event?.preventDefault()
    if (!ready || !draft.trim()) return
    send(draft)
    setDraft('')
  }
  const onKeyDown = event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }
  const copy = async (message, index) => {
    clearTimeout(copyTimer.current)
    setCopyNotice('')
    try {
      await navigator.clipboard.writeText(replyText(message))
      setCopied(index)
      setCopyNotice('Reply copied')
    } catch {
      setCopied(null)
      setCopyNotice('Unable to copy. You can select the reply text and copy it.')
    }
    copyTimer.current = setTimeout(() => { setCopied(null); setCopyNotice('') }, 3000)
  }

  return <section className="chat" aria-label="Conversation">
    <header className="chat__head">
      <span className="chat__mark"><Icon name="chat" size={20} /></span>
      <span className="chat__head-text"><strong>Your conversation</strong><span role="status">{subtitle}</span></span>
      {hpn && <button type="button" className={`topics-toggle${topicsInline ? ' topics-toggle--inline' : ''}`} ref={topicsButtonRef} data-testid="topics-toggle"
        aria-expanded={topicsOpen} aria-controls="conversation-topics" onClick={() => setTopicsOpen(open => !open)}>Topics <Icon name="chevronDown" size={14} /></button>}
      <button type="button" className="transcript-button" onClick={() => downloadTranscript(conv)} disabled={!conv.messages.length}
        data-testid="transcript" aria-label="Download conversation" title="Download conversation"><Icon name="download" size={19} /></button>
    </header>
    {hpn && <nav ref={topicsRef} id="conversation-topics" className={`explore${topicsInline ? ' explore--inline' : ''}${topicsOpen ? ' is-open' : ''}`} aria-label="Conversation starters">
      <div className="topic-grid">{TOPICS.map(topic => <button type="button" className="topic" key={topic.label} disabled={!ready} onClick={() => {
        send(topic.question)
        setTopicsOpen(false)
        if (topicsButtonRef.current?.getClientRects().length) topicsButtonRef.current.focus({ preventScroll: true })
      }}>
        <span className="topic__icon"><Icon name={topic.icon} size={20} /></span><strong>{topic.label}</strong>
      </button>)}</div>
    </nav>}
    <div className="conversation-history">
      <ol className={`messages ${conv.messages.length ? '' : 'messages--empty'}`} ref={listRef} onScroll={trackScroll}
        data-testid="messages" aria-label="Conversation messages" aria-live="polite" aria-relevant="additions text">
        {conv.messages.length === 0 && !busy && <li className="welcome-item">
          <HelperWelcome hpn={hpn} persona={conv.session?.persona} onChip={send} chipsDisabled={!ready} />
        </li>}
        {conv.messages.map((message, index) => <li key={`${message.turnId ?? 'past'}-${index}`}
          className={`msg ${message.role}${message.reply ? ' has-reply' : ''}${index === lastAssistant ? ' is-latest' : ''}`}
          data-role={message.role} data-turn={message.turnId} data-index={index}>
          <div className="msg__meta">
            <span className="msg__author">{message.role === 'user' ? 'You' : name}</span>
            {message.at && <time dateTime={new Date(message.at).toISOString()}>{TIME.format(message.at)}</time>}
          </div>
          {message.reply && (message.reply.text || message.reply.cards?.length)
            ? <HelperReply reply={message.reply} matched={message.matched} site={hpn?.site} showChips={index === lastAssistant}
              onChip={send} chipsDisabled={!ready} />
            : message.text ? <span className="text">{message.text}</span>
              : !message.error && <span className="reply-pending"><span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>Preparing an answer…</span>}
          {message.citations?.length > 0 && <ul className="sources" data-testid="sources">{message.citations.map(source =>
            <li key={source.n} title={source.snippet}>[{source.n}] {source.title}{source.section ? ` · ${source.section}` : ''}{source.page ? ` · p.${source.page}` : ''}</li>)}</ul>}
          {message.error && <span className="error" role="alert" data-testid="turn-error">{message.error}</span>}
          {message.role === 'assistant' && (message.done || message.error) && <div className="msg__actions">
            {replyText(message) && <button type="button" className="copy-reply" data-testid="copy-reply" onClick={() => copy(message, index)}
              aria-label={copied === index ? 'Reply copied' : 'Copy reply'}><Icon name={copied === index ? 'check' : 'copy'} size={13} />{copied === index ? 'Copied' : 'Copy'}</button>}
            {message.interrupted && <span className="tag">Reply stopped</span>}
          </div>}
        </li>)}
        {awaitingReply && <li className="reply-pending" data-testid="reply-pending" role="status">
          <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>{name} is thinking…
        </li>}
      </ol>
      {showLatest && <button type="button" className="jump-latest" data-testid="jump-latest" onClick={scrollToLatest}><Icon name="down" size={15} />Back to latest</button>}
    </div>
    <span className="copy-notice" role="status">{copyNotice}</span>
    <form className="composer" onSubmit={submit}>
      <label className="visually-hidden" htmlFor="conversation-input">Message</label>
      <div className="composer__field">
        <textarea id="conversation-input" ref={inputRef} data-testid="input" rows={1} value={draft} onChange={event => setDraft(event.target.value)}
          onKeyDown={onKeyDown} placeholder={conv.waiting ? 'Your conversation will start shortly…' : 'Ask about your event…'}
          autoComplete="off" enterKeyHint="send" maxLength={2000} aria-describedby="composer-hint" />
        {busy && <button type="button" className="stop-reply" data-testid="stop" onClick={conv.interrupt} aria-label="Stop reply" title="Stop reply"><Icon name="stop" size={17} /></button>}
        <button type="submit" data-testid="send" disabled={!draft.trim() || !ready} aria-label="Send message" title="Send message"><Icon name="send" size={20} /></button>
      </div>
      <div className="composer__meta" id="composer-hint"><span>{!ready ? subtitle : 'Enter to send · Shift + Enter for a new line'}</span>
        {draft.length >= 1800 ? <span className="character-count">{draft.length}/2000</span> : <span className="composer__trust">Answers from HPN</span>}
      </div>
    </form>
    {conv.role !== 'owner' && conv.persona?.remembered && <button type="button" className="forget-button" onClick={conv.forgetMemory} data-testid="forget-customer">Forget me</button>}
  </section>
}

function downloadTranscript(conv) {
  const name = conv.session?.persona || 'AI Agent'
  const lines = conv.messages.filter(message => replyText(message)).map(message =>
    `[${message.at ? TIME.format(message.at) : ''}] ${message.role === 'user' ? 'You' : name}: ${replyText(message)}${message.interrupted ? ' (interrupted)' : ''}`)
  const url = URL.createObjectURL(new Blob([`Conversation with ${name} — ${new Date().toLocaleString()}\n\n${lines.join('\n\n')}\n`], {type:'text/plain;charset=utf-8'}))
  const link = document.createElement('a')
  link.href = url
  link.download = `conversation-${new Date().toISOString().slice(0,16).replace(/[:T]/g,'-')}.txt`
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url),1000)
}
