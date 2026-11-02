import { useEffect, useState } from 'react'

// Mobile keyboards may resize only the visual viewport, leaving 100dvh unchanged.
// Keep the app and composer within that visible area without remounting the video.
export function useMobileViewport(appRef) {
  const [keyboardOpen, setKeyboardOpen] = useState(false)
  useEffect(() => {
    const viewport = window.visualViewport
    const mobile = window.matchMedia('(max-width: 760px)')
    let baseline = { width: window.innerWidth, height: window.innerHeight }
    let frame = 0
    const measure = () => {
      const app = appRef.current
      if (!app) return
      if (!mobile.matches || !viewport) {
        app.style.removeProperty('--visual-height')
        app.style.removeProperty('--visual-top')
        setKeyboardOpen(false)
        return
      }
      // Let visitors pinch-zoom normally rather than trying to resize the app to their zoom.
      if (Math.abs(viewport.scale - 1) > 0.02) return
      const editing = document.activeElement?.matches('textarea, input:not([type=checkbox])')
      if (baseline.width !== window.innerWidth) baseline = { width: window.innerWidth, height: window.innerHeight }
      if (!editing) baseline.height = Math.max(baseline.height, window.innerHeight, viewport.height)
      app.style.setProperty('--visual-height', `${viewport.height}px`)
      app.style.setProperty('--visual-top', `${viewport.offsetTop}px`)
      setKeyboardOpen(Boolean(editing && baseline.height - viewport.height > 100))
    }
    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(measure)
    }
    measure()
    viewport?.addEventListener('resize', schedule)
    viewport?.addEventListener('scroll', schedule)
    window.addEventListener('resize', schedule)
    document.addEventListener('focusin', schedule)
    document.addEventListener('focusout', schedule)
    return () => {
      cancelAnimationFrame(frame)
      viewport?.removeEventListener('resize', schedule)
      viewport?.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      document.removeEventListener('focusin', schedule)
      document.removeEventListener('focusout', schedule)
    }
  }, [appRef])
  return keyboardOpen
}
