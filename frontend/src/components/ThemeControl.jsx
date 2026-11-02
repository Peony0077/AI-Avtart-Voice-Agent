import { useEffect, useRef, useState } from 'react'
import Icon from './Icon.jsx'

const KEY = 'hpn-color-theme'
const THEMES = ['system', 'light', 'dark']
const readPreference = () => {
  try {
    const saved = localStorage.getItem(KEY)
    return THEMES.includes(saved) ? saved : 'system'
  } catch { return 'system' }
}

export default function ThemeControl() {
  const [preference, setPreference] = useState(readPreference)
  const [systemDark, setSystemDark] = useState(() => matchMedia('(prefers-color-scheme: dark)').matches)
  const detailsRef = useRef(null)
  const theme = preference === 'system' ? systemDark ? 'dark' : 'light' : preference

  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)')
    const update = () => setSystemDark(media.matches)
    const sync = event => { if (event.key === KEY || event.key === null) setPreference(readPreference()) }
    update()
    media.addEventListener('change', update)
    window.addEventListener('storage', sync)
    return () => {
      media.removeEventListener('change', update)
      window.removeEventListener('storage', sync)
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#17151b' : '#f8f7fa')
  }, [theme])

  useEffect(() => {
    const dismiss = event => {
      const details = detailsRef.current
      if (details && !details.contains(event.target)) details.open = false
    }
    const escape = event => {
      if (event.key === 'Escape' && detailsRef.current?.open) {
        detailsRef.current.open = false
        detailsRef.current.querySelector('summary')?.focus({ preventScroll: true })
      }
    }
    document.addEventListener('pointerdown', dismiss)
    document.addEventListener('focusin', dismiss)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', dismiss)
      document.removeEventListener('focusin', dismiss)
      document.removeEventListener('keydown', escape)
    }
  }, [])

  const choose = value => {
    setPreference(value)
    try { localStorage.setItem(KEY, value) } catch { /* Theme still works when storage is unavailable. */ }
    detailsRef.current.open = false
    detailsRef.current.querySelector('summary')?.focus({ preventScroll: true })
  }

  return <details className="theme-control" ref={detailsRef}>
    <summary aria-label="Color theme" title={`Color theme: ${preference}`} data-testid="theme-open"><Icon name={theme === 'dark' ? 'moon' : 'sun'} size={18} /></summary>
    <fieldset className="theme-control__options">
      <legend>Appearance</legend>
      {THEMES.map(value => <label key={value}>
        <input type="radio" name="color-theme" value={value} checked={preference === value} onChange={() => choose(value)} />
        {value === 'system' ? 'Use device setting' : value === 'light' ? 'Light' : 'Dark'}
      </label>)}
    </fieldset>
  </details>
}
