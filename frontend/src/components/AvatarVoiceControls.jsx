import Icon from './Icon.jsx'

// Overlay controls stay separate from the canvas so layout changes preserve media playback.
export default function AvatarVoiceControls({ conv, voice }) {
  const canTalk = conv.status === 'open' && !!conv.session && conv.session.voice_available !== false && !conv.waiting
  return canTalk ? (
    <div className="stage-voice">
      {voice.micHint && (
        <p className="stage-hint" data-testid="mic-hint">
          Your Bluetooth headset's microphone makes voices sound like a phone call.{' '}
          <button type="button" onClick={() => voice.useMic(voice.micHint.deviceId)}>Use {voice.micHint.label}</button>{' '}
          <button type="button" className="secondary" onClick={voice.dismissMicHint}>Keep</button>
        </p>
      )}
      {voice.error && <p className="stage-error" role="alert" data-testid="stage-error">{voice.error}</p>}
      {voice.active ? (
        <span className={`listening ${voice.speechDetected ? 'hot' : ''}`} data-testid="listening">
          <span className="dot" />
          {voice.speechDetected ? 'Hearing you…' : 'Listening – just speak'}
        </span>
      ) : (
        // the microphone only starts on a tap (browsers require it): make that impossible to miss
        <button type="button" className="talk" data-testid="talk" onClick={voice.start} disabled={voice.starting}>
          {!voice.starting && !voice.error && <Icon name="mic" size={18} />}
          {voice.starting ? 'Allow the microphone…' : voice.error ? 'Try again' : `Talk to ${conv.session?.persona ?? 'your event helper'}`}
        </button>
      )}
    </div>
  ) : null
}

