# HPN avatar interface

The interface is centered on talking to HPN's AI avatar. It uses a larger avatar stage,
a supporting conversation panel, opt-in microphone controls, and event shortcuts.
Customer links keep owner settings, leads, and diagnostics out of the customer interface.

## Research

Reviewed on November 3, 2026 using the connected MCP web tool and Playwright Chromium:

- [HPN homepage](https://h-townparty.com/): charcoal header, purple `#8d0bab` brand,
  Archivo Black headings, Montserrat body text, and event planning as the primary journey.
- [Event planner](https://h-townparty.com/plan-my-event/): direct route to building an event plan.
- [Nights out](https://h-townparty.com/night-out/): karaoke, Jukebox Bingo, and DJ nights.
- The homepage's Event helper: HPN information with an explicit distinction between the helper
  and the human team. The avatar keeps that distinction, with visible AI labeling and a team link.

Playwright opened the live homepage, captured its rendered appearance, inspected its fonts and
stylesheets, and opened its Event helper. DOM readiness was used because waiting for network
idle timed out on the live site.

## UI behavior

- The avatar canvas fits the full video frame without cutting off the face.
- Microphone access begins only after a user gesture. The button and status distinguish
  microphone off, listening, thinking, speaking, connecting, and waiting.
- Typed questions, welcome suggestions, and event shortcuts use the existing conversation flow.
- HPN replies retain structured cards, safe external links, and suggested follow-up questions.
- Waiting customers cannot submit questions or start their microphone.
- An unavailable avatar offers text conversation without showing a simulated live person.
- The owner menu contains Settings and Leads; customer sessions receive neither control.
- Narrow screens put the avatar before the conversation. “Type instead” focuses the composer.
- No new package prices, availability claims, or venue counts are embedded in the UI.

## Verification

```sh
cd tests/e2e
npm run test:ui
```

The isolated Playwright configuration starts/reuses Vite and intercepts WebSockets.
It does not connect to the GPU backend or run the production-data backup/restore hooks.
Checks cover 320, 390, 768, 1024, and 1440 pixel widths, video containment, no horizontal
overflow, questions and replies, cards and links, transcript download, microphone opt-in
and stop, waiting/unavailable states, and owner/customer control visibility.

Screenshots are written to `tests/e2e/results/ui/hpn-avatar-*.png`.
They use a local avatar reference frame when available; these screenshots verify the UI,
not generated speech, animation, or lip sync. Without a local reference image, a small
fixture image is used so CI does not need model or avatar downloads.

The production frontend build is checked separately with `npm run build` in `frontend/`.


## AI Agent wording, compact talk button, and long histories

Visible product labels and the helper introduction use **AI Agent**. The talk action is a
48 px pill with a lighter border, a single shadow, and an explicit keyboard focus ring.
The conversation panel stretches to its sibling stage on desktop; its scrollable message
list has a zero minimum height so growing history cannot grow the grid or push the footer.
On phones the chat retains a bounded height. The page no longer reserves flexible space
between the content and footer.

The regression suite also fills 80 exchanges (160 messages) at desktop and phone widths,
checking that document height, conversation height, and footer position stay unchanged.

Production builds create `/houston-party-network/index.html` alongside the root entry.
The same backend, static assets, WebSocket endpoint, tunnel hostname, and access tokens
continue to work. HPN tunnel scripts print this branded path automatically. The path
is a display/entry address; it does not replace the existing access-token requirement.


## Single-window layout

The introductory marketing section has been removed. The app now fills `100dvh`;
its header, shortcuts, and footer take only their required room, and the avatar/chat
area uses the remaining height. Video keeps its aspect ratio. On phones, the avatar
and chat share the available height vertically. Only chat history and open owner
dialogs/menus scroll. Diagnostics are inside the owner menu so they cannot extend
the page. The welcome adapts to the conversation panel height.

Playwright checks viewport containment at 1440×900, 1366×768, 1280×720, 1024×768,
768×1024, 390×844, 320×568, and 844×390, including visibility of the talk control,
composer, shortcuts, and footer. Long histories remain in the conversation panel.


## Full-width avatar stage

The latest interface removes the outer card borders, rounded frames, shadows, and
page gutters. The video fills the available stage from the header to the footer;
identity, status, and voice controls overlay the video. `object-fit: contain` retains
the complete source frame. The unframed conversation panel sits alongside it on
desktop and below it on phones. Event shortcuts are compact controls within the
conversation panel, freeing more height for the avatar. Browser checks verify
zero stage border/radius/shadow and full viewport width, alongside existing
single-window and long-history checks.

See [HPN_AGENT_DOMAIN.md](HPN_AGENT_DOMAIN.md) for setting up `agent.h-townparty.com`.


## Video-proportioned desktop columns

The desktop avatar column is now sized from the available height and the source
video aspect ratio, instead of using a wide flexible column. This removes unused
left/right bands for the square HPN video. The conversation expands across all
remaining width. The column is capped to leave at least 340 px for chat in narrower
desktop windows. Phone layouts keep the stacked arrangement and reserve at least
240 px for the conversation so the input remains usable. Video stays uncropped.
Playwright checks also cover 1920×1080 and 2560×1080 and measure the video-column
and conversation widths explicitly.


## Conversation interaction and spacing

The workspace, top bar, and footer use one centered content-width rule: a
1,240 px maximum with responsive side gutters of 32, 20, or 12 px. Their left
and right edges align at every tested viewport size. Conversation header, shortcuts, history, and composer share
one inset rule; wide panels center their content within a 760 px reading width.
The welcome is balanced within that space instead of hugging one side.

`ConversationPanel.jsx` owns the draft and conversation interactions:

- A bounded multiline textarea sends with Enter; Shift+Enter adds a new line.
  The existing 2,000-character server limit is reflected in the composer.
- Messages show author, AI labeling, and local timestamps.
- Finished replies can be copied, including their structured card details.
  Copy success and clipboard errors are announced without interrupting chat.
- Thinking/preparing status and Stop reply use the existing server state and
  interrupt message; no new backend protocol is introduced.
- Incoming text follows the latest answer until the visitor scrolls up. Reading
  earlier messages is preserved, and Back to latest resumes automatic following.
- Downloads retain the readable reply text and card details.

The Playwright UI suite has 25 checks covering these behaviors and existing
layout, voice opt-in, access-role visibility, and long-history constraints.


## Mobile conversation layout and viewport research

The mobile avatar is again a large, full-width video above the conversation. The
previous 136 px thumbnail strip and its expand/collapse control were removed.
The video row uses the source aspect ratio and available height, ranging from
220 px on short phones to 320 px on taller screens. At 390×844, the row is
approximately 305 px high; merely focusing the composer does not shrink it.
The same live canvas and media subscriptions remain mounted throughout.

Space now comes from the conversation interface:

- Messages use 13 px text with 1.6 line height, smaller bubble padding, and less
  space between paragraphs and messages. Desktop messages remain 14 px.
- The mobile header offers a Topics disclosure instead of a permanent shortcut
  row. All four shortcuts remain available. The overlay does not resize history;
  Escape, outside interaction, and choosing a topic close it. Keyboard focus
  returns to the disclosure after Escape or selection.
- The welcome heading and prompts have less decoration and spacing.
- The composer remains fixed below the independently scrolling history. Its
  textarea stays at 16 px; viewport zoom is unrestricted.

Research informed this choice; it does not prescribe these exact dimensions:

- [W3C Reflow guidance](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html)
  permits adjusting or relocating responsive content while retaining access to
  information and functionality. Moving secondary topic navigation into a
  disclosure saves persistent space while keeping all shortcuts available.
- [W3C Resize Text guidance](https://www.w3.org/WAI/WCAG22/Understanding/resize-text.html)
  explains why resizing text must remain possible. Smaller default message text
  is a space adjustment, not a substitute for reflow or permission to disable zoom.
- [Chrome viewport/keyboard behavior](https://developer.chrome.com/blog/viewport-resize-behavior)
  explains that a keyboard may resize only the visual viewport, leaving fixed
  content behind the keyboard.

The viewport meta tag requests `interactive-widget=resizes-content` where
supported. `useMobileViewport` also responds to visible height and vertical offset.
Only when an actual viewport reduction accompanies editing does the video row
adapt to 112–150 px, retaining its full-width layout and microphone controls.
This temporary adaptation reserves space for messages and the keyboard; closing
the keyboard restores the large avatar. Secondary footer content and the central
Talk CTA are hidden during keyboard input. Pinch zoom does not trigger viewport
adjustments. ResizeObserver follows the latest message as available height
changes, while preserving the position of readers who have scrolled up.

Validation: frontend production build and 33 Playwright UI checks passed. Mobile
cases include 320×568, 360×640, 390×844, and 430×932, large video dimensions,
message typography, topic disclosure/keyboard interaction, microphone opt-in,
long histories, and a simulated visual-viewport keyboard resize/pan. Headless
Chromium has no physical phone keyboard: the test exercises the browser API
boundary, rather than certifying real iOS or Android keyboard behavior.

The rebuilt public route also returned HTTP 200 in Playwright at 390×844 and
1440×900. The mobile avatar measured 366×305 px, leaving approximately 360 px for
message history. The desktop chat retained 60% of content width. Neither layout
had page overflow or browser errors. Public checks intercept the WebSocket and
use a local reference frame plus fixture messages; they verify the deployed UI,
not live model responses, generated animation, or lip sync.


## Equal desktop sections

Desktop columns now divide the shared content width equally: 50% for the avatar
and 50% for the conversation at all widths above 760 px, including short windows.
The full video remains uncropped within its section. Header/content/footer
alignment and the stacked mobile layout remain in place. Desktop browser checks
explicitly measure both sections against half the available content width.

## Light and dark appearance

The header sun/moon control offers Light, Dark, and Use device setting. Device
setting is the default and follows OS changes until a visitor selects an explicit
theme. The choice is saved under `hpn-color-theme` in local storage and synced
between tabs. If storage is unavailable, theme selection still works for the
current page. An early script applies the initial theme before CSS renders.

Semantic CSS variables cover page surfaces, primary/secondary text, accent text,
message bubbles, cards, borders, scrollbars, composer states, menus, footer, and
owner forms. Dark mode uses charcoal surfaces and lighter purple text; primary
HPN purple buttons keep white labels. The brand header and avatar overlays retain
their dark treatment. `color-scheme` and the browser theme-color metadata follow
the selected theme. Switching themes preserves the existing media canvas and
conversation connection. The desktop 50/50 split and large mobile avatar remain.

Validation: production build and all 40 Playwright checks passed. Theme cases
cover desktop/mobile light and dark rendering, at least 4.5:1 contrast for sampled
message text, timestamps, card accent text, input text and placeholders, saved
preferences after reload, device changes, and dark owner dialogs. These samples
do not constitute a complete accessibility audit. Theme screenshots use fixture
messages and the local avatar reference frame.

## Consistent interface icons

`Icon.jsx` now uses a single curated Lucide SVG family instead of approximate
handwritten paths. Icons share a 24 px artboard, rounded corners, 2 px strokes,
and `currentColor` to inherit their light/dark theme colors. Static imports include
only the icons used by the app. The dependency and ISC license are recorded in
the npm lockfile. See [Lucide React documentation](https://lucide.dev/guide/react).

The agent identity uses an audio waveform, event planning uses a calendar, the
composer uses an upward send arrow, and Topics uses a disclosure chevron. All
microphone, mute, keyboard, copy, download, theme, settings, location, and link
icons use the same component. Duplicate microphone/link SVG implementations and
the text arrow on welcome suggestions were removed. Decorative SVGs are hidden
from assistive technology; existing button labels remain the accessible names.
Topic icons have square slots and theme-aware accent colors. The first-message
mobile Topics control remains available while desktop starters can stay inline.

Validation: frontend production build and all 40 existing Playwright checks
passed. The rebuilt public route also passed at 1440×844 and 390×844 in both
themes: HTTP 200, 22 SVG icons per initial page, uniform stroke/artboard attributes,
no page overflow, and no browser errors. Screenshots use intercepted WebSockets
and a local avatar reference frame, preserving any real customer conversation.
