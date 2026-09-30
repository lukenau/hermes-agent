/**
 * Tests for electron/installs-ipc.ts.
 *
 * The IPC layer is exercised with a fake ipcMain and a fake runner, so these
 * tests need no electron and no child process.
 */

import assert from 'node:assert/strict'

import { afterEach, beforeEach, test } from 'vitest'

import { createInstallsNotice, type InstallsNotice, type InstallsRunOutcome, registerInstallsIpc } from './installs-ipc'

const noNotice = { take: (): null => null }

const SAMPLE_LIST = {
  current: 'abc',
  installs: [
    {
      id: 'abc',
      root: '/home/u/hermes-agent',
      steward: 'git',
      version: '1.2.3',
      sources: ['path'],
      current: true,
      package_full_name: null,
      removable: false,
      action: 'refuse',
      refusal: 'not removed: this install is running'
    }
  ],
  launchers: [],
  notice: { count: 0, dismissed: false }
}

type Handler = (event: unknown, payload?: unknown) => Promise<unknown>

function fakeIpcMain(): { handlers: Map<string, Handler>; handle: (channel: string, handler: Handler) => void } {
  const handlers = new Map<string, Handler>()

  return {
    handlers,
    handle: (channel, handler) => {
      handlers.set(channel, handler)
    }
  }
}

const call = (ipc: ReturnType<typeof fakeIpcMain>, channel: string, payload?: unknown): Promise<unknown> =>
  ipc.handlers.get(channel)!(null, payload)

test('registerInstallsIpc list parses the CLI JSON', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: `noise\n${JSON.stringify(SAMPLE_LIST, null, 2)}\n`, stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const list = (await call(ipc, 'hermes:installs:list')) as typeof SAMPLE_LIST | null

  assert.deepEqual(runs, [['installs', 'list', '--json']])
  assert.equal(list?.current, 'abc')
  assert.equal(list?.installs[0]?.removable, false)
})

test('registerInstallsIpc list returns null on a non-zero exit and bad JSON', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: () => Promise.resolve({ code: 1, stdout: '', stderr: 'boom' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  assert.equal(await call(ipc, 'hermes:installs:list'), null)

  const badIpc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: badIpc,
    runInstalls: () => Promise.resolve({ code: 0, stdout: 'not json', stderr: '' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  assert.equal(await call(badIpc, 'hermes:installs:list'), null)
})

test('registerInstallsIpc remove validates the id before building argv', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: 'Removed.', stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const bad = (await call(ipc, 'hermes:installs:remove', { id: 'rm -rf /' })) as { ok: boolean; error?: string }

  assert.equal(bad.ok, false)
  assert.equal(bad.error, 'invalid-id')
  assert.deepEqual(runs, [])

  const good = (await call(ipc, 'hermes:installs:remove', { id: 'abc123' })) as { ok: boolean }

  assert.equal(good.ok, true)
  assert.deepEqual(runs, [['installs', 'remove', 'abc123', '--yes']])
})

test('registerInstallsIpc remove surfaces the CLI refusal on exit 1', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: () => Promise.resolve({ code: 1, stdout: '', stderr: 'not removed: this install is running' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  const result = (await call(ipc, 'hermes:installs:remove', { id: 'abc' })) as { ok: boolean; message?: string }

  assert.equal(result.ok, false)
  assert.equal(result.message, 'not removed: this install is running')
})

test('registerInstallsIpc dismiss runs the dismiss argv', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: 'ok', stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const result = (await call(ipc, 'hermes:installs:dismiss')) as { ok: boolean }

  assert.equal(result.ok, true)
  assert.deepEqual(runs, [['installs', 'dismiss']])
})

// --- boot notice ---

const NOTICED = {
  current: 'abc',
  installs: [],
  launchers: [],
  notice: { count: 2, dismissed: false }
}

let notice: InstallsNotice
let signals: number
let debugs: string[]
let runnerOutcome: InstallsRunOutcome
let runnerShouldReject: boolean

const runner = (): Promise<InstallsRunOutcome> =>
  runnerShouldReject ? Promise.reject(new Error('spawn failed')) : Promise.resolve(runnerOutcome)

const check = (): void =>
  notice.check({
    runInstalls: runner,
    signalNotice: () => {
      signals += 1
    },
    logDebug: message => debugs.push(message)
  })

const wait = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

beforeEach(() => {
  notice = createInstallsNotice()
  signals = 0
  debugs = []
  runnerOutcome = { code: 0, stdout: JSON.stringify(NOTICED), stderr: '' }
  runnerShouldReject = false
})

afterEach(() => {
  runnerOutcome = { code: null, stdout: '', stderr: '' }
  runnerShouldReject = false
})

test('boot notice waits for the renderer: found with no window listening, taken once on mount', async () => {
  check()
  await wait(10)

  // Nothing listened when it was found (the renderer had not mounted). The pull still gets it.
  assert.equal(signals, 1)
  assert.deepEqual(notice.take(), { count: 2 })
  assert.equal(notice.take(), null)
  assert.deepEqual(debugs, [])
})

test('boot notice stays silent when dismissed, empty, failed, or malformed', async () => {
  for (const outcome of [
    { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 2, dismissed: true } }), stderr: '' },
    { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 0, dismissed: false } }), stderr: '' },
    { code: 1, stdout: '', stderr: 'boom' },
    { code: 0, stdout: 'half printed {', stderr: '' }
  ]) {
    runnerOutcome = outcome
    check()
    await wait(10)
  }

  assert.equal(signals, 0)
  assert.equal(notice.take(), null)
})

test('boot notice is found once per app launch, and a launch with none keeps checking', async () => {
  runnerOutcome = { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 0, dismissed: false } }), stderr: '' }
  check()
  await wait(10)
  assert.equal(signals, 0)

  runnerOutcome = { code: 0, stdout: JSON.stringify(NOTICED), stderr: '' }
  check()
  await wait(10)
  check()
  await wait(10)

  assert.equal(signals, 1)
})

test('boot notice failures are logged, never thrown', async () => {
  runnerShouldReject = true
  check()
  await wait(10)

  assert.equal(signals, 0)
  assert.equal(notice.take(), null)
  assert.equal(debugs.length, 1)
  assert.match(debugs[0]!, /spawn failed/)
})

test('registerInstallsIpc take-notice hands the renderer the pending notice', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: runner,
    notice: { take: () => ({ count: 3 }) },
    logDebug: () => undefined
  })

  assert.deepEqual(await call(ipc, 'hermes:installs:take-notice'), { count: 3 })
})
