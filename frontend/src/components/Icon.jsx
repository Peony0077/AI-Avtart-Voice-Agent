import {
  ArrowDown, ArrowRight, ArrowUp, ArrowUpRight, AudioLines, CalendarDays,
  Check, ChevronDown, Copy, Download, Heart, Keyboard, MapPin,
  MessageSquareText, Mic, MicOff, Moon, Music2, Settings, Sparkles,
  Square, Sun, Ticket, Volume2, VolumeX,
} from 'lucide-react'

// One icon family, 24 px artboards and rounded 2 px strokes throughout the UI.
// Static imports keep unused Lucide icons out of the production bundle.
const ICONS = {
  agent: AudioLines,
  arrow: ArrowRight,
  send: ArrowUp,
  down: ArrowDown,
  chevronDown: ChevronDown,
  external: ArrowUpRight,
  calendar: CalendarDays,
  check: Check,
  copy: Copy,
  sparkles: Sparkles,
  heart: Heart,
  music: Music2,
  ticket: Ticket,
  mic: Mic,
  micOff: MicOff,
  volume: Volume2,
  volumeOff: VolumeX,
  keyboard: Keyboard,
  chat: MessageSquareText,
  download: Download,
  stop: Square,
  pin: MapPin,
  settings: Settings,
  sun: Sun,
  moon: Moon,
}

export default function Icon({ name, size = 20 }) {
  const Symbol = ICONS[name]
  if (!Symbol) return null
  return <Symbol className="icon" data-icon={name} size={size} strokeWidth={2}
    aria-hidden="true" focusable="false" />
}
