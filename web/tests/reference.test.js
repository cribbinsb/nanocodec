// Pre-encoded reference streams (reference.js, view-settings.js): the generated
// manifest is complete and every stream can be matched live by the other view;
// the loader picks and decodes the right frames on the shared timeline.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { CLIPS } from '../clips.js';
import { fetchStream, framesToDecode, referenceStream, streamFrameAtTick, streamGroups } from '../reference.js';
import { matchProblems, matchingSettings } from '../view-settings.js';
import { tickStep } from '../timeline.js';

const directory = new URL('../streams-ref/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', directory)));
// Clips whose terms allow shipping re-encodes (web/clips/ATTRIBUTION.md): no
// Netflix (CC BY-NC-ND: no derivatives) and no SVT Multi Format. MOTChallenge
// (CC BY-NC-SA) streams must carry their licence notice.
const SHIPPABLE = /^(xiph-|tos-|bbb-|mot(17|20)-|svt-(waterfall|midnight-sun|smoke-sauna)$|netflix-food-market$)/;
// GitHub Pages refuses files over 100 MB; the whole set stays far below it.
const MAX_SET_BYTES = 100 * 1024 * 1024;

test('every manifest stream exists, is described and can be matched live', () => {
  assert.ok(manifest.streams.length >= 34);
  let total = 0;
  for (const entry of manifest.streams) {
    assert.deepEqual(matchProblems(entry, CLIPS.map(c => c.id)), [], entry.id);
    assert.match(entry.clip, SHIPPABLE, entry.id);
    if (entry.clip.startsWith('mot')) assert.match(entry.notice, /CC BY-NC-SA 3\.0.*motchallenge\.net.*same CC BY-NC-SA 3\.0/, entry.id);
    // Netflix FoodMarket only (owner's decision): the full credit and licence, and what the stream is.
    if (entry.clip.startsWith('netflix')) assert.match(entry.notice, /NETFLIX INC\. 2015.*CC BY-NC-ND 4\.0.*non-commercial codec comparison/, entry.id);
    assert.equal(statSync(new URL(entry.file, directory)).size, entry.bytes, entry.id);
    for (const field of ['command', 'label']) assert.ok(entry[field], `${entry.id}: ${field}`);
    assert.ok(entry.version.version, `${entry.id}: version`);
    assert.ok(entry.actualKbps > 0 && entry.frames * tickStep(entry.fps) === entry.loopTicks, entry.id);
    total += entry.bytes;
  }
  assert.ok(total < MAX_SET_BYTES, `${total} bytes`);
});

test('the set is varied: content groups, encoders, sizes, frame rates and rates', () => {
  const distinct = key => new Set(manifest.streams.map(key)).size;
  assert.ok(distinct(e => CLIPS.find(c => c.id === e.clip).group) >= 4);
  assert.ok(distinct(e => e.encoder) >= 8);
  assert.ok(distinct(e => e.height) >= 4);
  assert.ok(distinct(e => e.fps) >= 3);
});

test('every clip offers at least three streams, some at one matched size and rate across encoders', () => {
  const groups = streamGroups(manifest.streams);
  for (const [clip, entries] of groups) {
    assert.ok(entries.length >= 3, `${clip}: ${entries.length} streams`);
    assert.ok(new Set(entries.map(e => e.codec)).size >= 2, `${clip}: one codec`);
  }
  // Same clip, size, frame rate and target across encoders: the best side-by-side sets.
  const matched = new Map();
  for (const e of manifest.streams) {
    const key = `${e.clip} ${e.width}x${e.height} ${e.fps} ${e.targetKbps}`;
    matched.set(key, new Set([...(matched.get(key) ?? []), e.codec]));
  }
  assert.ok([...matched.values()].filter(codecs => codecs.size >= 3).length >= 3);
  // The picker lists each clip once, entries of an encoder together.
  assert.equal(groups.length, new Set(manifest.streams.map(e => e.clip)).size);
});

test('supplied streams say what they are and match a live view with one key frame per loop', () => {
  const supplied = manifest.streams.filter(e => e.supplied);
  assert.equal(supplied.length, 6);
  for (const entry of supplied) {
    assert.equal(entry.label, 'BWC SOC');
    assert.equal(entry.command, 'not available (hardware encoder)');
    assert.match(entry.provenance, /Not made by any tool in this repository/);
    assert.match(entry.rateControl, /EVBR/);
    assert.match(entry.sourceNote, /dB lower/);
    assert.equal(entry.keyFrames, 1);
    // One key frame in the 10 s loop: the live view's key interval becomes 10 s.
    assert.equal(matchingSettings(entry).gopSeconds * entry.fps, entry.frames);
  }
  // Software encodes of the same clip, size, rate and bitrates, one key frame each.
  for (const encoder of ['x264-medium', 'openh264', 'x265-medium']) {
    const runs = manifest.streams.filter(e => e.clip === 'netflix-food-market' && e.encoder === encoder);
    assert.deepEqual(runs.map(e => e.targetKbps).sort(), [1000, 1500, 2000], encoder);
    for (const e of runs) assert.ok(e.keyFrames === 1 && e.width === 1280 && e.fps === 30 && e.rateControl, e.id);
  }
});

test('matching sets the stream\'s size, rate, buffer and key interval', () => {
  const entry = manifest.streams[0];
  const settings = matchingSettings(entry);
  assert.equal(settings.width, entry.width); assert.equal(settings.fps, entry.fps);
  assert.equal(settings.targetKbps, entry.targetKbps); assert.equal(settings.rateMode, 'cbr');
  assert.equal(settings.codec, entry.codec);
  assert.ok(matchProblems({ ...entry, fps: 25 }, [entry.clip]).includes('25 fps is not offered'));
  assert.deepEqual(matchProblems({ ...entry, clip: 'netflix-tango' }, [entry.clip]), ['clip netflix-tango is not a hosted sample clip']);
});

test('a stream shows its frame for each tick it codes and catches up on skipped ones', () => {
  const stream = { step: 2, loopTicks: 300 };
  assert.equal(streamFrameAtTick(stream, 4), 2);
  assert.equal(streamFrameAtTick(stream, 5), null);
  assert.equal(streamFrameAtTick(stream, 302), 1);
  assert.deepEqual(framesToDecode(null, 0), [0]);
  assert.deepEqual(framesToDecode(3, 6), [4, 5, 6]);
  // After a loop the decoder restarts from the key frame.
  assert.deepEqual(framesToDecode(148, 1), [0, 1]);
});

test('the loader checks the frame count and reports fetch progress and errors', async () => {
  const entry = manifest.streams[0];
  const bytes = readFileSync(new URL(entry.file, directory));
  const progress = [];
  const response = { ok: true, body: new Blob([bytes]).stream() };
  const fetched = await fetchStream(entry, fraction => progress.push(fraction), async () => response);
  assert.equal(progress.at(-1), 1);
  const stream = referenceStream(entry, fetched);
  assert.equal(stream.packets.length, entry.frames);
  assert.throws(() => referenceStream({ ...entry, frames: entry.frames + 1 }, fetched));
  await assert.rejects(fetchStream(entry, () => {}, async () => ({ ok: false, status: 404 })), /HTTP 404/);
  await assert.rejects(fetchStream(entry, () => {}, async () => { throw new TypeError('offline'); }), /network error/);
  const short = { ok: true, body: new Blob([bytes.subarray(0, 100)]).stream() };
  await assert.rejects(fetchStream(entry, () => {}, async () => short), /truncated/);
});
