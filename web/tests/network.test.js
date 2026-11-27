// The in-page network emulator (network.js): seeded and repeatable, mean loss
// rate and burst length as set, loss scoped by layer, a bandwidth cap's queue,
// delay, jitter without reordering, and the feedback and playout delays.
import test from 'node:test';
import assert from 'node:assert/strict';
import { NetworkEmulator, DEFAULT_NETWORK, feedbackDelayMs, inLossScope, playoutDelayMs, roundTripMs } from '../network.js';

const PACKETS = 20000;
const packet = () => new Uint8Array(1000);
const base = { spatial: 0, temporal: 0 }, top = { spatial: 0, temporal: 1 };

/** Send PACKETS packets one ms apart; the fates in order. */
function fates(settings, layerOf = () => base) {
  const network = new NetworkEmulator(settings);
  return Array.from({ length: PACKETS }, (_, i) => network.send(packet(), layerOf(i), i));
}

test('the same seed loses the same packets', () => {
  const settings = { ...DEFAULT_NETWORK, lossPercent: 5 };
  assert.deepEqual(fates(settings), fates(settings));
  assert.notDeepEqual(fates(settings), fates({ ...settings, seed: 2 }));
});

test('random and burst loss reach the mean rate; bursts have the mean length', () => {
  for (const lossModel of ['random', 'burst']) {
    const lost = fates({ ...DEFAULT_NETWORK, lossModel, lossPercent: 5, burstLength: 4 });
    const rate = lost.filter(f => f === 'lost').length / PACKETS;
    assert.ok(Math.abs(rate - 0.05) < 0.012, `${lossModel}: ${rate}`);
    const bursts = lost.join(' ').split(/sent/).map(run => run.trim().split(' ').filter(Boolean).length).filter(Boolean);
    const meanBurst = bursts.reduce((a, b) => a + b, 0) / bursts.length;
    if (lossModel === 'burst') assert.ok(meanBurst > 3 && meanBurst < 5, `burst ${meanBurst}`);
    else assert.ok(meanBurst < 1.2, `random ${meanBurst}`);
  }
});

test('loss applies only to the chosen layers', () => {
  assert.ok(inLossScope('all', base) && inLossScope('base', base) && !inLossScope('enhancement', base));
  assert.ok(inLossScope('enhancement', top) && !inLossScope('base', top));
  const lost = fates({ ...DEFAULT_NETWORK, lossPercent: 20, lossScope: 'enhancement' }, i => (i % 2 ? top : base));
  assert.ok(lost.every((fate, i) => i % 2 || fate === 'sent'));
  assert.ok(lost.some((fate, i) => i % 2 && fate === 'lost'));
});

test('packets arrive after the delay and jitter, in order without reordering', () => {
  const network = new NetworkEmulator({ ...DEFAULT_NETWORK, lossPercent: 0, delayMs: 50, jitterMs: 20 });
  const sent = Array.from({ length: 100 }, (_, i) => { const p = new Uint8Array([i]); network.send(p, base, i * 5); return p; });
  assert.deepEqual(network.due(49), []);
  const arrived = network.due(1000);
  assert.deepEqual(arrived, sent);
});

test('a bandwidth cap queues packets and drops them past the queue', () => {
  // 1028 wire bytes at 800 kbps take ~10.3 ms each: sending one per ms overfills a 100 ms queue.
  const lost = fates({ ...DEFAULT_NETWORK, lossPercent: 0, delayMs: 0, jitterMs: 0, bandwidthKbps: 800, queueMs: 100 });
  const overflowed = lost.filter(f => f === 'overflow').length;
  assert.ok(overflowed > PACKETS * 0.85, `${overflowed}`);
});

test('feedback takes the round trip less the forward delay; the jitter buffer waits past the jitter', () => {
  const settings = { ...DEFAULT_NETWORK, delayMs: 40, jitterMs: 10, rttMs: null };
  assert.equal(roundTripMs(settings), 80);
  assert.equal(feedbackDelayMs(settings), 40);
  assert.equal(feedbackDelayMs({ ...settings, rttMs: 200 }), 160);
  assert.ok(playoutDelayMs(settings) > 50);
});
