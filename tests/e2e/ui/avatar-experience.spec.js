import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const photo = path.join(root, 'data/avatars/dev_girl/photo.png')
// Real local avatar when available; CI needs no downloaded avatar assets.
const image = fs.existsSync(photo) ? fs.readFileSync(photo) : Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC1sAAAAASUVORK5CYII=', 'base64')
const session = {
  type: 'session', session_id: 'ui-test', role: 'customer', persona: 'Emma',
  profile: 'hpn', avatar_available: true, voice_available: true,
  avatar: { size: [512, 512], fps: 25 },
  hpn: { site: 'https://h-townparty.com', chips: ['Weddings', "What's on tonight?"], phone_display: '(832) 295-0852' },
}

async function fixture(page, options = {}) {
  const received = []
  let sendToPage
  await page.routeWebSocket('**/ws?**', ws => {
    const send = data => ws.send(JSON.stringify(data))
    sendToPage = send
    ws.onMessage(raw => {
      if (typeof raw !== 'string') return
      const msg = JSON.parse(raw)
      received.push(msg)
      if (msg.type === 'hello') {
        if (options.waiting) { send({type: 'waiting', position: 2}); return }
        send({ ...session, ...options.session })
        send({type: 'state', state: 'LISTENING'})
        if (options.session?.avatar_available !== false) {
          for (let i = 0; i < 7; i++) {
            const header = Buffer.alloc(16)
            header[0] = 3 // real protocol: idle PNG/JPEG packet
            header.writeUInt32LE(i, 8)
            ws.send(Buffer.concat([header, image]))
          }
        }
      }
      if (msg.type === 'get_persona') send({type: 'persona', name: 'Emma', remembered: false, voices: ['af_heart']})
      if (msg.type === 'user_text') {
        send({type: 'user_message', turn_id: 1, text: msg.text, source: 'text'})
        send({type: 'state', state: 'THINKING'})
        send({type: 'helper_reply', turn_id: 1, reply: {
          text: 'Let’s find the right fit for your celebration.',
          cards: [{title: 'Wedding options', subtitle: 'Explore HPN’s packages', path: '/weddings/'}],
          links: [{label: 'Plan your wedding', url: 'https://h-townparty.com/contact-us/?event=wedding'}],
          chips: ['What does the package include?'],
        }})
        send({type: 'llm_delta', turn_id: 1, text: 'Let’s find the right fit for your celebration.'})
        send({type: 'assistant_done', turn_id: 1, text: 'Let’s find the right fit for your celebration.', interrupted: false})
        send({type: 'state', state: 'LISTENING'})
      }
      if (msg.type === 'voice' && msg.enabled) send({type: 'vad', event: 'start'})
    })
  })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page).toHaveTitle('Houston Party Network · AI Agent')
  if (!options.waiting) await expect(page.getByText('Ready to chat', {exact: true})).toBeVisible()
  return { received, errors, send: data => sendToPage(data) }
}

for (const theme of ['light', 'dark']) {
  for (const width of [1440, 390]) {
    test(`${theme} theme renders readable chat and cards at ${width}px`, async ({page}) => {
      await page.setViewportSize({width, height:844})
      await page.emulateMedia({colorScheme:theme})
      const {errors} = await fixture(page)
      await expect(page.locator('html')).toHaveAttribute('data-theme',theme)
      await expect(page.locator('html')).toHaveCSS('color-scheme',theme)
      await page.getByTestId('input').fill('Tell me about weddings')
      await page.getByTestId('send').click()
      await expect(page.locator('.helper-card')).toBeVisible()
      const contrast = await page.evaluate(() => {
        const rgb = value => value.match(/[\d.]+/g).slice(0,3).map(Number)
        const luminance = color => rgb(color).map(value => {
          const s = value / 255
          return s <= .04045 ? s / 12.92 : ((s + .055) / 1.055) ** 2.4
        }).reduce((sum,value,i) => sum + value * [.2126,.7152,.0722][i],0)
        const ratio = (foreground,background) => {
          const a=luminance(foreground), b=luminance(background)
          return (Math.max(a,b)+.05)/(Math.min(a,b)+.05)
        }
        const color = (selector,property,pseudo) => getComputedStyle(document.querySelector(selector),pseudo).getPropertyValue(property)
        const chat=color('.chat','background-color'), card=color('.helper-card','background-color')
        return {
          message:ratio(color('.msg.assistant','color'),chat),
          metadata:ratio(color('.msg__meta time','color'),chat),
          card:ratio(color('.helper-card__subtitle','color'),card),
          input:ratio(color('.composer textarea','color'),color('.composer__field','background-color')),
          placeholder:ratio(color('.composer textarea','color','::placeholder'),color('.composer__field','background-color')),
        }
      })
      for (const value of Object.values(contrast)) expect(value).toBeGreaterThanOrEqual(4.5)
      await expect(page.getByTestId('input')).toBeInViewport({ratio:1})
      await expect(page.getByTestId('theme-open')).toBeInViewport({ratio:1})
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth && document.documentElement.scrollHeight<=innerHeight)).toBe(true)
      await page.screenshot({path:`results/ui/hpn-theme-${theme}-${width}.png`,fullPage:true})
      expect(errors).toEqual([])
    })
  }
}

test('theme choice persists without replacing the avatar or reconnecting the conversation', async ({page}) => {
  await page.emulateMedia({colorScheme:'light'})
  const {received} = await fixture(page)
  await page.getByTestId('input').fill('My birthday party')
  await page.getByTestId('send').click()
  await page.getByTestId('avatar-canvas').evaluate(el=>{window.__originalCanvas=el})
  await page.getByTestId('theme-open').click()
  await page.getByRole('radio',{name:'Dark',exact:true}).check()
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await expect(page.getByTestId('theme-open')).toBeFocused()
  await expect(page.locator('.msg.user')).toContainText('My birthday party')
  expect(await page.getByTestId('avatar-canvas').evaluate(el=>el===window.__originalCanvas)).toBe(true)
  expect(received.filter(message=>message.type==='hello')).toHaveLength(1)
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await page.getByTestId('theme-open').click()
  await expect(page.getByRole('radio',{name:'Dark',exact:true})).toBeChecked()
  await page.keyboard.press('Escape')
  await expect(page.locator('.theme-control')).not.toHaveAttribute('open','')
})

test('device theme follows changes until overridden and can be restored', async ({page}) => {
  await page.emulateMedia({colorScheme:'dark'})
  await fixture(page)
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await page.emulateMedia({colorScheme:'light'})
  await expect(page.locator('html')).toHaveAttribute('data-theme','light')
  await page.getByTestId('theme-open').click()
  await page.getByRole('radio',{name:'Dark',exact:true}).check()
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await page.emulateMedia({colorScheme:'dark'})
  await page.emulateMedia({colorScheme:'light'})
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await page.getByTestId('theme-open').click()
  await page.getByRole('radio',{name:'Use device setting'}).check()
  await expect(page.locator('html')).toHaveAttribute('data-theme','light')
})

test('dark theme also covers owner tools and form controls', async ({page}) => {
  await page.emulateMedia({colorScheme:'dark'})
  await fixture(page,{session:{role:'owner'}})
  await page.getByLabel('Owner tools',{exact:true}).click()
  await expect(page.locator('.owner-menu__items')).toHaveCSS('background-color','rgb(29, 26, 33)')
  await page.getByTestId('settings-open').click()
  await expect(page.getByTestId('settings')).toHaveCSS('background-color','rgb(29, 26, 33)')
  await expect(page.getByTestId('persona-name')).toHaveCSS('color','rgb(238, 231, 241)')
})

for (const [width, height] of [[2560, 1080], [1920, 1080], [1440, 900], [1366, 768], [1280, 720], [1024, 768], [768, 1024], [390, 844], [320, 568], [844, 390]]) {
  test(`avatar, conversation and controls fit one window at ${width}x${height}`, async ({ page }) => {
    await page.setViewportSize({width, height})
    const {errors} = await fixture(page)
    await expect(page.getByTestId('talk')).toBeVisible()
    await expect(page.locator('.intro')).toHaveCount(0)
    await expect(page.locator('h1')).toHaveText('Houston Party Network AI Agent')
    await expect(page.locator('.stage--playing')).toBeVisible()
    await expect(page.locator('.stage-placeholder')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true)
    await expect(page.getByTestId('talk')).toBeInViewport({ratio: 1})
    await expect(page.getByTestId('input')).toBeInViewport({ratio: 1})
    await expect(page.locator('.site-footer')).toBeInViewport({ratio: 1})
    if (width <= 760) await page.getByTestId('topics-toggle').click()
    await expect(page.locator('.topic').last()).toBeInViewport({ratio: 1})
    if (width <= 760) await page.keyboard.press('Escape')
    const stage = await page.locator('.stage').boundingBox()
    const chat = await page.locator('.chat').boundingBox()
    const canvas = await page.getByTestId('avatar-canvas').boundingBox()
    expect(canvas.height).toBeLessThanOrEqual(stage.height + 1)
    expect(canvas.y + canvas.height).toBeLessThanOrEqual(stage.y + stage.height + 1)
    const frame = await page.locator('.stage-col').evaluate(el => {
      const css = getComputedStyle(el)
      const main = document.querySelector('.experience').getBoundingClientRect()
      const layout = document.querySelector('.layout').getBoundingClientRect()
      const header = document.querySelector('.topbar__inner').getBoundingClientRect()
      const footer = document.querySelector('.site-footer__inner').getBoundingClientRect()
      return {radius:css.borderRadius, border:css.borderWidth, shadow:css.boxShadow, width:main.width, layoutWidth:layout.width, left:layout.left, right:layout.right, headerLeft:header.left, headerRight:header.right, headerWidth:header.width, footerLeft:footer.left, footerRight:footer.right}
    })
    expect(frame.radius).toBe('0px')
    expect(frame.border).toBe('0px')
    expect(frame.shadow).toBe('none')
    expect(Math.abs(frame.width - frame.headerWidth)).toBeLessThanOrEqual(1)
    expect(Math.abs(frame.left - frame.headerLeft)).toBeLessThanOrEqual(1)
    expect(Math.abs(frame.right - frame.headerRight)).toBeLessThanOrEqual(1)
    expect(Math.abs(frame.left - frame.footerLeft)).toBeLessThanOrEqual(1)
    expect(Math.abs(frame.right - frame.footerRight)).toBeLessThanOrEqual(1)
    await expect(page.getByTestId('avatar-canvas')).toHaveCSS('object-fit', 'contain')
    if (width > 760) {
      // Both desktop sections occupy exactly half the shared content width.
      expect(Math.abs(stage.width - frame.layoutWidth / 2)).toBeLessThanOrEqual(1)
      expect(Math.abs(chat.width - frame.layoutWidth / 2)).toBeLessThanOrEqual(1)
      expect(Math.abs(stage.width - chat.width)).toBeLessThanOrEqual(1)
    }
    await page.screenshot({path: `results/ui/hpn-avatar-${width}.png`, fullPage: true})
    expect(errors).toEqual([])
  })
}

test('typing, suggested questions, HPN answer cards and download work', async ({page}) => {
  const {received, errors} = await fixture(page)
  await page.getByRole('button', {name: 'Type instead'}).click()
  await expect(page.getByTestId('input')).toBeFocused()
  await page.getByTestId('input').fill('Tell me about weddings')
  await page.getByTestId('send').click()
  await expect(page.locator('li[data-role="user"]')).toContainText('Tell me about weddings')
  await expect(page.locator('.helper-card')).toContainText('Wedding options')
  await expect(page.getByRole('link', {name:'Plan your wedding'})).toHaveAttribute('href', 'https://h-townparty.com/contact-us/?event=wedding')
  await expect(page.getByRole('link', {name:'Plan your wedding'})).toHaveAttribute('rel', 'noopener noreferrer')
  await page.getByRole('button', {name:'What does the package include?'}).click()
  expect(received.filter(m => m.type === 'user_text').map(m => m.text)).toContain('What does the package include?')
  const downloaded = page.waitForEvent('download')
  await page.getByTestId('transcript').click()
  expect((await downloaded).suggestedFilename()).toMatch(/^conversation-.*\.txt$/)
  await page.screenshot({path:'results/ui/hpn-avatar-reply.png',fullPage:true})
  expect(errors).toEqual([])
})

test('topic shortcuts send questions through the conversation', async ({page}) => {
  const {received} = await fixture(page)
  await page.getByRole('button', {name:'Nights out', exact:true}).click()
  await expect.poll(() => received.some(m => m.type === 'user_text' && m.text === "What's on tonight?")).toBe(true)
})

test('waiting customers cannot send or start voice', async ({page}) => {
  const {received} = await fixture(page, {waiting:true})
  await expect(page.getByTestId('waiting')).toBeVisible()
  await expect(page.getByTestId('talk')).toHaveCount(0)
  await expect(page.getByTestId('voice-start')).toBeDisabled()
  await page.getByTestId('input').fill('Hello')
  await expect(page.getByTestId('send')).toBeDisabled()
  expect(received.some(m => m.type === 'user_text')).toBe(false)
})

test('avatar unavailable leaves an honest fallback and working text', async ({page}) => {
  await fixture(page, {session:{avatar_available:false, voice_available:false}})
  await expect(page.getByText('The avatar is unavailable.', {exact:false})).toBeVisible()
  await expect(page.getByTestId('talk')).toHaveCount(0)
  await page.getByTestId('input').fill('Can I ask about an event?')
  await page.getByTestId('send').click()
  await expect(page.locator('li[data-role="assistant"]')).toContainText('celebration')
})

test('microphone is opt-in and can be stopped', async ({page}) => {
  const {received, send} = await fixture(page)
  expect(received.some(m => m.type === 'voice' && m.enabled)).toBe(false)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('voice-stop')).toBeVisible()
  await expect(page.getByTestId('talk')).toHaveCount(0)
  send({type: 'waiting', position: 1})
  await expect(page.getByTestId('voice-stop')).toBeEnabled()
  await page.getByTestId('voice-stop').click()
  send({...session, session_id: 'promoted'})
  await expect(page.getByTestId('talk')).toBeVisible()
  expect(received.some(m => m.type === 'voice' && !m.enabled)).toBe(true)
})

test('owner controls are separate from the customer experience', async ({page}) => {
  await fixture(page)
  await expect(page.getByLabel('Owner tools', {exact:true})).toHaveCount(0)
})


test('owner menu exposes settings and leads only after opening', async ({page}) => {
  await fixture(page, {session: {role: 'owner'}})
  await expect(page.getByTestId('settings-open')).not.toBeVisible()
  await page.getByLabel('Owner tools', {exact: true}).click()
  await expect(page.getByTestId('settings-open')).toBeVisible()
  await expect(page.getByTestId('leads-open')).toBeVisible()
  await page.getByTestId('settings-open').click()
  await expect(page.getByTestId('settings')).toBeVisible()
})

for (const width of [1440, 390]) {
  test(`long history scrolls inside the chat without moving the footer at ${width}px`, async ({page}) => {
    await page.setViewportSize({width, height: 1000})
    const {send, errors} = await fixture(page)
    const measure = () => page.evaluate(() => {
      const chat = document.querySelector('.chat').getBoundingClientRect()
      const footer = document.querySelector('.site-footer').getBoundingClientRect()
      const experience = document.querySelector('.experience').getBoundingClientRect()
      const layout = document.querySelector('.layout').getBoundingClientRect()
      return {height: document.documentElement.scrollHeight, chat: chat.height,
        footer: footer.top + scrollY, trailingSpace: experience.bottom - layout.bottom}
    })
    const before = await measure()
    for (let i = 1; i <= 80; i++) {
      send({type:'user_message', turn_id:i, source:'text', text:`Question ${i}: tell me more about planning an event.`})
      send({type:'llm_delta', turn_id:i, text:'Here is some information about your event. '.repeat(10)})
      send({type:'assistant_done', turn_id:i, interrupted:false})
    }
    await expect(page.locator('.msg')).toHaveCount(160)
    const after = await measure()
    expect(Math.abs(after.height - before.height)).toBeLessThanOrEqual(2)
    expect(Math.abs(after.chat - before.chat)).toBeLessThanOrEqual(2)
    expect(Math.abs(after.footer - before.footer)).toBeLessThanOrEqual(2)
    expect(after.trailingSpace).toBeLessThanOrEqual(16)
    const scroll = await page.getByTestId('messages').evaluate(el => ({scroll:el.scrollHeight, client:el.clientHeight, top:el.scrollTop}))
    expect(scroll.scroll).toBeGreaterThan(scroll.client * 5)
    expect(scroll.scroll - scroll.client - scroll.top).toBeLessThanOrEqual(2)
    await page.screenshot({path:`results/ui/hpn-agent-long-chat-${width}.png`,fullPage:true})
    expect(errors).toEqual([])
  })
}


test('owner diagnostics stay in the tools menu without extending the page', async ({page}) => {
  await page.setViewportSize({width: 1280, height: 720})
  const {send} = await fixture(page, {session: {role: 'owner'}})
  send({type:'metrics', turn_id:1, ts:Date.now(), ms_from_t0:{llm_done:100}})
  await page.getByLabel('Owner tools', {exact:true}).click()
  await expect(page.getByTestId('metrics')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(true)
  await expect(page.getByTestId('input')).toBeInViewport({ratio:1})
})


test('Enter sends and Shift+Enter adds a new line', async ({page}) => {
  const {received} = await fixture(page)
  const field = page.getByTestId('input')
  await field.fill('A birthday party')
  await field.press('Shift+Enter')
  await field.pressSequentially('For forty guests')
  expect(received.some(message => message.type === 'user_text')).toBe(false)
  await field.press('Enter')
  expect(received.filter(message => message.type === 'user_text').at(-1).text).toBe('A birthday party\nFor forty guests')
  await expect(field).toHaveValue('')
})

test('completed replies have timestamps and copy their text and card details', async ({page, context}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await fixture(page)
  await page.getByTestId('input').fill('Wedding information')
  await page.getByTestId('send').click()
  await expect(page.locator('.msg time')).toHaveCount(2)
  await page.getByTestId('copy-reply').click()
  await expect(page.getByTestId('copy-reply')).toHaveText('Copied')
  const text = await page.evaluate(() => navigator.clipboard.readText())
  expect(text).toContain('Let’s find the right fit')
  expect(text).toContain('Wedding options')
})

test('new replies respect reading position and Back to latest resumes following', async ({page}) => {
  const {send} = await fixture(page)
  for (let i = 1; i <= 40; i++) {
    send({type:'user_message', turn_id:i, source:'text', text:`Question ${i}`})
    send({type:'llm_delta', turn_id:i, text:'An event answer with enough detail to scroll. '.repeat(8)})
    send({type:'assistant_done', turn_id:i, interrupted:false})
  }
  await expect(page.locator('.msg')).toHaveCount(80)
  const list = page.getByTestId('messages')
  await list.evaluate(element => {element.scrollTop = 0})
  await expect(page.getByTestId('jump-latest')).toBeVisible()
  send({type:'llm_delta', turn_id:41, text:'A new answer arriving while you read.'})
  await expect(page.locator('.msg')).toHaveCount(81)
  expect(await list.evaluate(element => element.scrollTop)).toBe(0)
  await page.getByTestId('jump-latest').click()
  await expect(page.getByTestId('jump-latest')).toHaveCount(0)
  await expect.poll(() => list.evaluate(element => element.scrollHeight - element.clientHeight - element.scrollTop)).toBeLessThanOrEqual(2)
})

test('reply progress and Stop reply use the existing conversation controls', async ({page}) => {
  const {send, received} = await fixture(page)
  send({type:'user_message', turn_id:77, source:'text', text:'Help with my party'})
  send({type:'state', state:'THINKING'})
  await expect(page.getByTestId('reply-pending')).toBeVisible()
  await expect(page.getByRole('status').filter({hasText:'Emma is preparing your answer'})).toBeVisible()
  await page.getByTestId('stop').click()
  expect(received.some(message => message.type === 'interrupt')).toBe(true)
  send({type:'state', state:'LISTENING'})
  await expect(page.getByTestId('reply-pending')).toHaveCount(0)
})

test('clipboard errors are visible and do not break the conversation', async ({page}) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', {value:{writeText:async () => {throw new Error('clipboard blocked')}}}))
  const {errors} = await fixture(page)
  await page.getByTestId('input').fill('Tell me more')
  await page.getByTestId('send').click()
  await page.getByTestId('copy-reply').click()
  await expect(page.getByRole('status').filter({hasText:'Unable to copy'})).toBeVisible()
  await expect(page.getByTestId('input')).toBeEnabled()
  expect(errors).toEqual([])
})

for (const [width, height] of [[320,568],[360,640],[390,844],[430,932]]) {
  test(`mobile ${width}x${height} keeps a large live avatar and readable conversation`, async ({page}) => {
    await page.setViewportSize({width,height})
    const {send} = await fixture(page)
    const chat = await page.locator('.chat').boundingBox()
    const avatar = await page.locator('.stage-col').boundingBox()
    expect(avatar.height).toBeGreaterThanOrEqual(219)
    expect(avatar.width).toBeGreaterThanOrEqual(width - 25)
    expect(chat.height).toBeGreaterThan(height * .44)
    await expect(page.getByRole('navigation', {name:'Conversation starters'})).not.toBeVisible()
    for(let i=1;i<=12;i++) {
      send({type:'user_message',turn_id:i,source:'text',text:`Event question ${i}`})
      send({type:'llm_delta',turn_id:i,text:'We can help you explore your event options. Tell us the date and the type of celebration.'})
      send({type:'assistant_done',turn_id:i,interrupted:false})
    }
    await expect(page.locator('.msg')).toHaveCount(24)
    const readingHeight = await page.getByTestId('messages').evaluate(el=>el.clientHeight)
    expect(readingHeight).toBeGreaterThan(height * .27)
    await expect(page.locator('.msg.assistant').last()).toHaveCSS('font-size','13px')
    expect(await page.getByTestId('messages').evaluate(el=>el.scrollWidth<=el.clientWidth)).toBe(true)
    await expect(page.getByTestId('avatar-canvas')).toBeInViewport({ratio:1})
    await expect(page.getByTestId('input')).toBeInViewport({ratio:1})
    await expect(page.locator('.msg.assistant').last()).toContainText('event options')
    await page.screenshot({path:`results/ui/hpn-mobile-conversation-${width}.png`,fullPage:true})
  })
}

test('mobile keeps the large canvas when focusing the conversation', async ({page}) => {
  await page.setViewportSize({width:390,height:844})
  await fixture(page)
  await page.getByTestId('avatar-canvas').evaluate(el=>{window.__originalCanvas=el})
  const before = await page.locator('.stage-col').boundingBox()
  expect(before.height).toBeGreaterThan(300)
  await page.getByRole('button', {name:'Type instead'}).click()
  await expect(page.getByTestId('input')).toBeFocused()
  expect((await page.locator('.stage-col').boundingBox()).height).toBe(before.height)
  expect(await page.getByTestId('avatar-canvas').evaluate(el=>el===window.__originalCanvas)).toBe(true)
  await page.screenshot({path:'results/ui/hpn-mobile-large-avatar.png',fullPage:true})
})

test('mobile topics disclose without shrinking chat, dismiss, and send a question', async ({page}) => {
  await page.setViewportSize({width:390,height:844})
  const {received} = await fixture(page)
  const toggle = page.getByTestId('topics-toggle')
  const topics = page.getByRole('navigation',{name:'Conversation starters'})
  const before = await page.getByTestId('messages').boundingBox()
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-expanded','true')
  await expect(topics).toBeVisible()
  expect((await page.getByTestId('messages').boundingBox()).height).toBe(before.height)
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button',{name:'Plan an event',exact:true})).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(toggle).toBeFocused()
  await expect(topics).not.toBeVisible()
  await toggle.click()
  await page.getByTestId('input').click()
  await expect(topics).not.toBeVisible()
  await toggle.click()
  await page.getByRole('button',{name:'Weddings',exact:true}).click()
  await expect(topics).not.toBeVisible()
  await expect(toggle).toBeFocused()
  expect(received.some(message=>message.type==='user_text'&&message.text==='Tell me about weddings')).toBe(true)
})

test('mobile microphone controls remain usable over the large avatar', async ({page}) => {
  await page.setViewportSize({width:390,height:844})
  const {received} = await fixture(page)
  await page.getByTestId('talk').click()
  await expect(page.getByTestId('listening')).toBeVisible()
  await expect(page.getByTestId('voice-stop')).toBeInViewport({ratio:1})
  await page.getByTestId('voice-stop').click()
  await expect(page.getByTestId('talk')).toBeInViewport({ratio:1})
  expect(received.some(message=>message.type==='voice'&&message.enabled)).toBe(true)
})

test('visual-viewport keyboard resizing keeps the input, newest message, and avatar visible', async ({page}) => {
  await page.setViewportSize({width:390,height:844})
  // Desktop headless browsers have no native mobile keyboard. Model the browser API boundary:
  // layout viewport stays 844px tall while the keyboard reduces/pans the visual viewport.
  await page.addInitScript(()=>{
    let height=844,top=0
    const viewport=new EventTarget()
    Object.defineProperties(viewport,{
      height:{get:()=>height},width:{get:()=>390},offsetTop:{get:()=>top},scale:{get:()=>1},
    })
    Object.defineProperty(window,'visualViewport',{configurable:true,value:viewport})
    window.__resizeVisualViewport=(nextHeight,nextTop=0)=>{
      height=nextHeight;top=nextTop
      viewport.dispatchEvent(new Event('resize'))
      viewport.dispatchEvent(new Event('scroll'))
    }
  })
  const {send,errors} = await fixture(page)
  for(let i=1;i<=20;i++){
    send({type:'user_message',turn_id:i,source:'text',text:`Question ${i}`})
    send({type:'llm_delta',turn_id:i,text:'Here are the details of your event. '.repeat(6)})
    send({type:'assistant_done',turn_id:i,interrupted:false})
  }
  await expect(page.locator('.msg')).toHaveCount(40)
  await page.getByTestId('input').focus()
  await page.evaluate(()=>window.__resizeVisualViewport(440,50))
  await expect(page.locator('.app')).toHaveClass(/keyboard-open/)
  await expect(page.locator('.site-footer')).not.toBeVisible()
  const bounds=await page.evaluate(()=>{
    const field=document.querySelector('.composer').getBoundingClientRect()
    const avatar=document.querySelector('canvas').getBoundingClientRect()
    const list=document.querySelector('.messages')
    return{bottom:field.bottom,avatarTop:avatar.top,avatarBottom:avatar.bottom,avatarWidth:avatar.width,readingHeight:list.clientHeight,gap:list.scrollHeight-list.clientHeight-list.scrollTop}
  })
  expect(bounds.bottom).toBeLessThanOrEqual(490)
  expect(bounds.avatarTop).toBeGreaterThanOrEqual(50)
  expect(bounds.avatarBottom).toBeLessThanOrEqual(490)
  expect(bounds.avatarWidth).toBeGreaterThan(360)
  expect(bounds.readingHeight).toBeGreaterThan(130)
  expect(bounds.gap).toBeLessThanOrEqual(2)
  await page.screenshot({path:'results/ui/hpn-mobile-keyboard.png',fullPage:true})
  await page.getByTestId('input').blur()
  await page.evaluate(()=>window.__resizeVisualViewport(844,0))
  await expect(page.locator('.app')).not.toHaveClass(/keyboard-open/)
  await expect(page.locator('.site-footer')).toBeVisible()
  expect(errors).toEqual([])
})
