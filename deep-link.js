// Links between the demo (index.html) and the results pages (results/): the
// URL parameters each page reads, built and parsed in one place so the two
// directions stay in step (tests/deep-link.test.js, tests/deep-links.py).
//
// Demo:    ?codec=av1&preset=fast&size=1280x720&fps=30&kbps=600[&stream=ID][&clip=ID]
//          view B runs live nanocodec at these settings; view A shows pre-encoded
//          stream ID (a reference encoder's point), or else the original source.
//          ?scene=ID (alone or with the above): the source is test scene ID (scene.js).
// Results: ?codec=av1&preset=fast   the overview filtered to that codec and preset,
//          with that preset's point selected.

/**
 * The ladder presets (codecs.js PRESETS) that the results charts have points for.
 * A chart published before 2026-11-21 calls today's Slower `slow`.
 */
export const CHART_PRESETS = ['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast'];
const DEFAULT_PRESET = 'medium';
const SIZE_PATTERN = /^(\d+)x(\d+)$/;
const ID_PATTERN = /^[a-z0-9][a-z0-9-]*$/;
const CODEC_PATTERN = /^[a-z0-9]+$/;

/**
 * A chart reference family's pre-encoded streams in the demo's manifest
 * (streams-ref/manifest.json `encoder`): chart family names and the manifest's differ.
 */
export function manifestEncoder(family) {
  const [base] = family.split('~');
  const rules = [
    [/^x26[45]-[a-z]+$/, (name) => name],
    [/^svtav1-m\d+$/, (name) => name],
    [/^libvpx-vp9-cpu(\d+)$/, (_, cpu) => `vpx-rt${cpu}`],
    [/^libaom-av1-cpu(\d+)$/, (_, cpu) => `aom-rt${cpu}`],
    [/^openh264-baseline$/, () => 'openh264'],
  ];
  for (const [pattern, name] of rules) {
    const match = base.match(pattern);
    if (match) return name(...match);
  }
  return null;
}

/** The demo's query string for these settings (all optional but codec). */
export function demoQuery({ codec, preset = DEFAULT_PRESET, width, height, fps, kbps, stream, clip }) {
  const params = new URLSearchParams({ codec, preset });
  if (width && height) params.set('size', `${width}x${height}`);
  if (fps) params.set('fps', String(fps));
  if (kbps) params.set('kbps', String(kbps));
  if (stream) params.set('stream', stream);
  if (clip) params.set('clip', clip);
  return `?${params}`;
}

/** The settings a demo link asks for, or null when the URL carries none (or invalid ones). */
export function parseDemoQuery(search) {
  const params = new URLSearchParams(search);
  const codec = params.get('codec');
  if (!codec || !CODEC_PATTERN.test(codec)) return null;
  const preset = params.get('preset') || DEFAULT_PRESET;
  if (!CHART_PRESETS.includes(preset)) return null;
  const link = { codec, preset };
  const size = params.get('size')?.match(SIZE_PATTERN);
  if (size) { link.width = Number(size[1]); link.height = Number(size[2]); }
  for (const key of ['fps', 'kbps']) {
    const value = Number(params.get(key));
    if (Number.isFinite(value) && value > 0) link[key] = value;
  }
  for (const key of ['stream', 'clip']) {
    const value = params.get(key);
    if (value && ID_PATTERN.test(value)) link[key] = value;
  }
  return link;
}

/** The test scene a demo link asks for (`scene`, a scene.js id), or null. */
export function parseSceneQuery(search) {
  const scene = new URLSearchParams(search).get('scene');
  return scene && ID_PATTERN.test(scene) ? scene : null;
}

/** The results overview's query string for a codec and preset (the demo's names). */
export function resultsQuery({ codec, preset = DEFAULT_PRESET }) {
  return `?${new URLSearchParams({ codec, preset })}`;
}

/** The codec and preset a results link asks for, or null. */
export function parseResultsQuery(search) {
  const params = new URLSearchParams(search);
  const codec = params.get('codec');
  const preset = params.get('preset') || DEFAULT_PRESET;
  if (!codec || !CODEC_PATTERN.test(codec) || !CHART_PRESETS.includes(preset)) return null;
  return { codec, preset };
}

/** The chart family of our encoder at a codec and preset (nceval names: Medium is the bare codec). */
export function chartFamily({ codec, preset }) {
  return preset === DEFAULT_PRESET ? `nanocodec-${codec}` : `nanocodec-${codec}@${preset}`;
}
