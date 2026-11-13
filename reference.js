// Pre-encoded reference streams: sample clips coded offline by third-party
// encoders (x264, openh264, x265, SVT-AV1, libaom, libvpx, NVENC) in the
// configurations the project's charts use, which a view can play instead of a
// live nanocodec encode. tools/web/make-reference-streams.py makes the files and
// the manifest in streams-ref/ (encoder, version, exact command line, clip,
// size, frame rate, target and actual bitrate); the page keeps no list of its
// own. A stream's file is fetched only when a view selects it.
import { ivfPackets } from './streams.js';
import { tickStep } from './timeline.js';

export const MANIFEST_URL = new URL('./streams-ref/manifest.json', import.meta.url);

let manifest = null;

/** The manifest's stream records, fetched once. */
export function loadManifest(fetcher = fetch) {
  manifest ??= fetcher(MANIFEST_URL).then(response => {
    if (!response.ok) throw new Error(`No pre-encoded stream list (${response.status})`);
    return response.json();
  }).then(data => data.streams).catch(error => { manifest = null; throw error; });
  return manifest;
}

/**
 * The streams grouped for the picker: by clip, in manifest order, and within a
 * clip by encoder, then size, frame rate and bitrate. Returns [[clip, entries]].
 */
export function streamGroups(streams) {
  const groups = new Map();
  for (const entry of streams) groups.set(entry.clip, [...(groups.get(entry.clip) ?? []), entry]);
  const order = (a, b) => a.label.localeCompare(b.label) || b.height - a.height || b.fps - a.fps || a.targetKbps - b.targetKbps;
  return [...groups].map(([clip, entries]) => [clip, entries.sort(order)]);
}

/** One line naming a stream in a picker. */
export function streamTitle(entry) {
  return `${entry.label} · ${entry.width}×${entry.height} · ${entry.fps} fps · ${entry.targetKbps} kbps`;
}

/**
 * What a view needs to show a stream: its file and packets (one per coded frame), the
 * tick step of its frame rate (timeline.js) and the loop length it was made for.
 */
export function referenceStream(entry, bytes) {
  const packets = ivfPackets(bytes);
  if (packets.length !== entry.frames) throw new Error(`${entry.file}: ${packets.length} frames, the manifest says ${entry.frames}`);
  return { entry, bytes, packets, step: tickStep(entry.fps), loopTicks: entry.loopTicks };
}

/** The frame of a stream shown at `tick`, or null when the stream has no frame on that tick. */
export function streamFrameAtTick(stream, tick) {
  const local = tick % stream.loopTicks;
  return local % stream.step === 0 ? local / stream.step : null;
}

/**
 * The frames to decode to show frame `target` after `last` (null: nothing yet):
 * every frame since `last`, as each predicts from the one before; after a loop,
 * from the stream's first (key) frame.
 */
export function framesToDecode(last, target) {
  const first = last === null || target <= last ? 0 : last + 1;
  return Array.from({ length: target - first + 1 }, (_, i) => first + i);
}

/**
 * Fetch a stream's bytes, reporting `onProgress(fraction)` while they arrive;
 * rejects with a readable message on a network or HTTP error.
 */
export async function fetchStream(entry, onProgress, fetcher = fetch) {
  const url = new URL(`./streams-ref/${entry.file}`, import.meta.url);
  let response;
  try { response = await fetcher(url); } catch { throw new Error(`Could not fetch ${entry.file}: network error`); }
  if (!response.ok) throw new Error(`Could not fetch ${entry.file} (HTTP ${response.status})`);
  const bytes = new Uint8Array(entry.bytes);
  let received = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > bytes.length) throw new Error(`${entry.file} is larger than the manifest says`);
    bytes.set(value, received); received += value.length;
    onProgress(received / bytes.length);
  }
  if (received !== bytes.length) throw new Error(`${entry.file} is truncated (${received} of ${bytes.length} bytes)`);
  return bytes;
}
