import Icon from './Icon.jsx'

// Houston Party Network mode: the Event helper's answer drawn the way the website's helper draws it -
// paragraphs with **bold**, cards (packages, tonight's nights), link buttons and suggested questions.
// Everything is built as React nodes from the server's strings: nothing is parsed as HTML.

// Only these open from the page: a link from the server that is anything else (javascript:, data:) is
// not drawn at all.
const SAFE_URL = /^(https?:|tel:|mailto:)/i

function hrefOf(item, site) {
  const raw = item?.url || (item?.path && site ? joinUrl(site, item.path) : '')
  return raw && SAFE_URL.test(raw) ? raw : null
}

function joinUrl(site, path) {
  try {
    return new URL(path, site).href
  } catch {
    return ''
  }
}

// A phone number or an email opens in this tab (the phone app or the mail app takes it); a web page in a new one.
const external = (href) => /^https?:/i.test(href)
const linkProps = (href) => (external(href) ? { href, target: '_blank', rel: 'noopener noreferrer' } : { href })

// "**bold**" -> <strong>; a blank line starts a paragraph, a single line break stays a line break.
export function FormattedText({ text, className = 'helper-p' }) {
  const paragraphs = String(text ?? '').replace(/\r\n?/g, '\n').split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean)
  return paragraphs.map((p, i) => (
    <p key={i} className={className}>
      {p.split('\n').map((line, j) => (
        <span key={j}>
          {j > 0 && <br />}
          {inline(line)}
        </span>
      ))}
    </p>
  ))
}

function inline(line) {
  // split with a capture group: the odd parts are what stood between ** and **; an unpaired ** stays as text
  return line.split(/\*\*(.+?)\*\*/g).map((part, k) => (k % 2 ? <strong key={k}>{part}</strong> : part))
}

export function Chips({ chips, onPick, disabled, label = 'Suggested questions' }) {
  if (!chips?.length) return null
  return (
    <div className="helper-chips" role="group" aria-label={label}>
      {chips.map((chip) => (
        <button key={chip} type="button" className="chip" onClick={() => onPick(chip)} disabled={disabled}>
          {chip}
          {label === 'Ways to start' && <Icon name="arrow" size={14} />}
        </button>
      ))}
    </div>
  )
}

// Call, email and the contact form: on a phone these are never among the buttons left out.
const isContact = (href) => /^(tel:|mailto:)/i.test(href) || /\/contact-us\//.test(href)

// A suggested question that only repeats a button above it ("Request a song" twice) is left out.
function withoutLinked(chips, links) {
  const labels = new Set(links.map((l) => l.label.trim().toLowerCase()))
  return (chips ?? []).filter((chip) => !labels.has(chip.trim().toLowerCase()))
}

export default function HelperReply({ reply, matched, site, showChips, onChip, chipsDisabled }) {
  const cards = (reply.cards ?? []).filter((c) => c?.title)
  const links = (reply.links ?? []).map((l) => ({ ...l, href: hrefOf(l, site) })).filter((l) => l.label && l.href)
  return (
    <div className="helper-reply">
      {matched && <p className="helper-matched">Answering: “{matched}”</p>}
      <FormattedText text={reply.text} />
      {cards.length > 0 && (
        <ul className="helper-cards">
          {cards.map((c, i) => {
            const href = hrefOf(c, site)
            const body = (
              <>
                <span className="helper-card__title">{c.title}</span>
                {c.subtitle && <span className="helper-card__subtitle">{c.subtitle}</span>}
                {c.detail && <span className="helper-card__detail">{c.detail}</span>}
              </>
            )
            return (
              <li key={i}>
                {href ? <a className="helper-card helper-card--link" {...linkProps(href)}>{body}</a>
                  : <div className="helper-card">{body}</div>}
              </li>
            )
          })}
        </ul>
      )}
      {links.length > 0 && (
        <div className="helper-links">
          {links.map((l, i) => (
            <a key={`${l.href}-${i}`} className={`btn btn-sm ${i === 0 ? 'btn-primary' : 'btn-secondary'}${isContact(l.href) ? ' is-contact' : ''}`} {...linkProps(l.href)}>
              {l.label}
              {external(l.href) && <Icon name="external" size={14} />}
              {external(l.href) && <span className="visually-hidden"> (opens in a new tab)</span>}
            </a>
          ))}
        </div>
      )}
      {showChips && <Chips chips={withoutLinked(reply.chips, links)} onPick={onChip} disabled={chipsDisabled} />}
    </div>
  )
}

// A conversational introduction; the suggestions use the same server flow as typed questions.
export function HelperWelcome({ hpn, persona, onChip, chipsDisabled }) {
  return (
    <div className="welcome">
      <span className="welcome__symbol"><Icon name="agent" size={28} /></span>
      <p className="welcome__greeting">{persona ? `Hi, I'm ${persona}.` : 'A warm welcome to HPN.'}</p>
      <h2 className="welcome__title">Let’s make it<br /> memorable.</h2>
      <p className="welcome__lead">{hpn ? 'A wedding, a big birthday, or just a great night out. Tell me what you have in mind.' : 'Ask a question by voice or type a message to get started.'}</p>
      {hpn && <div className="welcome__prompts"><span>NOT SURE WHERE TO START?</span>
        <Chips chips={['How much is a wedding?', "What's on tonight?"]} onPick={onChip} disabled={chipsDisabled} label="Ways to start" />
      </div>}
    </div>
  )
}
