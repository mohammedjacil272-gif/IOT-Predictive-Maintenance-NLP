import assert from 'node:assert/strict';
import { test } from 'node:test';
import { startBackend } from '../.test-build/backendStartup.js';

const tick = () => new Promise(resolve => setImmediate(resolve));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function deps(overrides = {}) {
  const events = [];
  const ready = [];
  const readyTimeouts = [];
  const errors = [];
  let handler;
  let disposed = 0;
  const listenerGate = deferred();
  const base = {
    isDevelopment: false,
    listen: async callback => {
      events.push('listen');
      handler = callback;
      await listenerGate.promise;
      return () => { disposed += 1; events.push('dispose'); };
    },
    invoke: async () => { events.push('invoke'); return { port: 48123, token: 'test-token' }; },
    waitUntilReady: async (url, _signal, timeoutMs) => { events.push(`ready:${url}`); ready.push(url); readyTimeouts.push(timeoutMs); },
    onReady: (url, token) => { events.push(`onReady:${url}:${token ?? ''}`); },
    onError: message => { errors.push(message); events.push(`error:${message}`); },
  };
  return {
    events, ready, readyTimeouts, errors,
    get handler() { return handler; },
    get disposed() { return disposed; },
    releaseListener: () => listenerGate.resolve(),
    deps: { ...base, ...overrides },
  };
}

test('registers listener before invoking backend_config and preserves event/command ordering', async () => {
  const h = deps();
  const stop = startBackend(h.deps);
  await tick();
  assert.deepEqual(h.events, ['listen']);
  h.handler({ port: 49001, token: 'event-token' });
  h.releaseListener();
  await tick();
  await tick();
  assert.deepEqual(h.events, ['listen', 'invoke', 'ready:http://127.0.0.1:48123', 'onReady:http://127.0.0.1:48123:test-token']);
  assert.equal(h.ready[0], 'http://127.0.0.1:48123');
  assert.equal(h.readyTimeouts[0], 60000);
  stop();
  assert.equal(h.disposed, 1);
});

test('passes a bounded configurable readiness timeout to the health poller', async () => {
  const h = deps({ readinessTimeoutMs: 90000 });
  const stop = startBackend(h.deps);
  await tick();
  h.releaseListener();
  await tick();
  await tick();
  assert.equal(h.readyTimeouts[0], 90000);
  stop();
});

test('cancellation aborts readiness, suppresses callbacks, and cleans up the listener', async () => {
  const h = deps({
    waitUntilReady: async (_url, signal) => await new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    }),
  });
  const stop = startBackend(h.deps);
  await tick();
  h.releaseListener();
  await tick();
  stop();
  await tick();
  assert.equal(h.disposed, 1);
  assert.equal(h.events.some(x => x.startsWith('onReady:')), false);
  assert.equal(h.errors.length, 0);
});

test('listener promise resolving after cancellation is still disposed and command is never invoked', async () => {
  const h = deps();
  const stop = startBackend(h.deps);
  await tick();
  stop();
  h.releaseListener();
  await tick();
  assert.equal(h.disposed, 1);
  assert.equal(h.events.includes('invoke'), false);
});

test('uses dynamic command configuration over an earlier event recovery signal', async () => {
  const h = deps({ invoke: async () => ({ port: 49222, token: 'command-token' }) });
  const stop = startBackend(h.deps);
  await tick();
  h.handler({ port: 49221, token: 'event-token' });
  h.releaseListener();
  await tick();
  await tick();
  assert.equal(h.ready.at(-1), 'http://127.0.0.1:49222');
  assert.equal(h.events.at(-1), 'onReady:http://127.0.0.1:49222:command-token');
  stop();
});

test('reports timeout/readiness failures without leaking an error after cancellation', async () => {
  const h = deps({ waitUntilReady: async () => { throw new Error('readiness timeout after retries'); } });
  const stop = startBackend(h.deps);
  await tick();
  h.releaseListener();
  await tick();
  await tick();
  assert.deepEqual(h.errors, ['Backend startup failed: readiness timeout after retries']);
  stop();

  const cancelled = deps({ waitUntilReady: async () => { throw new Error('timeout'); } });
  const cancel = startBackend(cancelled.deps);
  await tick();
  cancel();
  cancelled.releaseListener();
  await tick();
  assert.deepEqual(cancelled.errors, []);
});

test('rejects invalid configuration and tolerates duplicate Strict Mode-style initialization', async () => {
  const invalid = deps({ invoke: async () => ({ port: 0, token: '' }) });
  const stopInvalid = startBackend(invalid.deps);
  await tick();
  invalid.releaseListener();
  await tick();
  await tick();
  assert.match(invalid.errors[0], /connection details/);
  stopInvalid();

  const first = deps();
  const stopFirst = startBackend(first.deps);
  await tick();
  stopFirst();
  first.releaseListener();
  await tick();
  const second = deps();
  const stopSecond = startBackend(second.deps);
  await tick();
  second.releaseListener();
  await tick();
  await tick();
  assert.equal(first.events.includes('onReady:http://127.0.0.1:48123:test-token'), false);
  assert.equal(second.events.includes('onReady:http://127.0.0.1:48123:test-token'), true);
  stopSecond();
});
