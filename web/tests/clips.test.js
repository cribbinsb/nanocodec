// The hosted clip files' video bitrates (mp4.js trackKbps), which the views compare
// their target with (view-settings.js overSourceShare): checked against the files'
// own media data, and the note's threshold; no pre-encoded stream codes a clip file at
// more than half its bitrate (tools/web/recut-clips.py makes the files heavy enough).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CLIPS } from '../clips.js';
import { readMp4, trackKbps } from '../mp4.js';
import { SOURCE_RATE_SHARE, overSourceShare } from '../view-settings.js';

const manifest = JSON.parse(readFileSync(new URL('../streams-ref/manifest.json', import.meta.url)));
// clips.js VARIANT_HEIGHTS: a view of height h reads the smallest file at least h tall.
const variant = height => (height <= 360 ? 360 : 720);

const clipFile = (id, height) => readFileSync(new URL(`../clips/${id}-${height}p.mp4`, import.meta.url));

/** The payload bytes of the file's top-level `mdat` boxes: a video-only file's samples. */
function mdatBytes(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let total = 0;
  for (let at = 0; at < bytes.length;) {
    const size = view.getUint32(at), type = String.fromCharCode(...bytes.subarray(at + 4, at + 8));
    if (type === 'mdat') total += size - 8;
    at += size;
  }
  return total;
}

test('a clip file\'s bitrate is its media data over its playing time', () => {
  for (const clip of CLIPS) {
    for (const height of [360, 720]) {
      const bytes = clipFile(clip.id, height), track = readMp4(bytes);
      const expected = mdatBytes(bytes) * 8 / (track.samples.length / clip.fps) / 1000;
      assert.ok(Math.abs(trackKbps(track, clip.fps) - expected) < 1e-6 * expected, `${clip.id} ${height}p`);
    }
  }
});

test('a target over half the source rate is flagged; an unknown source rate is not', () => {
  assert.equal(overSourceShare(500, 1000), false);
  assert.equal(overSourceShare(510, 1000), true);
  assert.equal(overSourceShare(5000, null), false);
});

test('every pre-encoded stream codes its clip file at most at half the file\'s bitrate', () => {
  for (const entry of manifest.streams) {
    const clip = CLIPS.find(c => c.id === entry.clip);
    const kbps = trackKbps(readMp4(clipFile(entry.clip, variant(entry.height))), clip.fps);
    assert.ok(entry.targetKbps <= SOURCE_RATE_SHARE * kbps, `${entry.id}: ${entry.targetKbps} kbps on a ${kbps.toFixed(0)} kbps file`);
  }
});
