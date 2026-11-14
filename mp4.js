// A minimal MP4 (ISO BMFF) reader for the page's own clips: the first video
// track's coded samples, in decode order, with their presentation times and key
// flags, and the decoder configuration (codec string and avcC/hvcC/av1C
// record) that WebCodecs' VideoDecoder needs (clip-source.js). It reads what
// ffmpeg writes for the hosted clips (ISO/IEC 14496-12: moov/trak/mdia/minf/stbl
// with stsd, stts, ctts, stsc, stsz, stco/co64, stss, and an edit list) and
// returns null for anything else (other codecs, fragmented files), so the caller
// can fall back.

const BOX_HEADER = 8, LARGE_SIZE_FIELD = 8;
// Containers whose children are boxes.
const CONTAINERS = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl', 'edts']);
// Sample entry header before its child boxes (ISO/IEC 14496-12 §12.1.3, VisualSampleEntry).
const VISUAL_SAMPLE_ENTRY_BYTES = 78;
// H.264 sample entries and their configuration box (ISO/IEC 14496-15 §5.3): the
// hosted clips are H.264; other codecs make the reader return null.
const AVC_ENTRIES = new Set(['avc1', 'avc3']);
// An edit's media time of -1 is an empty edit (ISO/IEC 14496-12 §8.6.6).
const EMPTY_EDIT = -1;

/** The boxes in `view` from `start` to `end`: [{ type, start, end }] (payload range). */
function boxes(view, start, end) {
  const out = [];
  for (let at = start; at + BOX_HEADER <= end;) {
    let size = view.getUint32(at), header = BOX_HEADER;
    const type = String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + at + 4, 4));
    if (size === 1) { size = Number(view.getBigUint64(at + BOX_HEADER)); header += LARGE_SIZE_FIELD; } else if (size === 0) size = end - at;
    if (size < header || at + size > end) break;
    out.push({ type, start: at + header, end: at + size });
    at += size;
  }
  return out;
}

/** The first box of `type` along `path` of nested types below `parent`. */
function find(view, parent, ...path) {
  let box = parent;
  for (const type of path) {
    box = boxes(view, box.start, box.end).find(b => b.type === type);
    if (!box) return null;
  }
  return box;
}

/** Full box payload after version and flags. */
const fullBox = box => ({ ...box, start: box.start + 4 });

function table(view, box, entryBytes, read) {
  const b = fullBox(box), count = view.getUint32(b.start), out = [];
  for (let i = 0; i < count; i++) out.push(read(b.start + 4 + i * entryBytes));
  return out;
}

/** The hex codec string WebCodecs expects for an avcC record (RFC 6381: avc1.PPCCLL). */
function avcCodec(record) {
  return `avc1.${[record[1], record[2], record[3]].map(b => b.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * The first video track of an MP4 file, or null if it is not one this reader handles:
 * `{ codec, description, width, height, timescale, samples: [{ offset, size, pts, duration, key }] }`
 * with samples in decode order and `pts` in timescale units, edit-list shift applied.
 */
export function readMp4(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const moov = boxes(view, 0, bytes.length).find(b => b.type === 'moov');
  if (!moov) return null;
  for (const trak of boxes(view, moov.start, moov.end).filter(b => b.type === 'trak')) {
    const hdlr = find(view, trak, 'mdia', 'hdlr');
    if (!hdlr || String.fromCharCode(...bytes.subarray(hdlr.start + 8, hdlr.start + 12)) !== 'vide') continue;
    const mdhd = fullBox(find(view, trak, 'mdia', 'mdhd'));
    const version = bytes[mdhd.start - 4];
    const timescale = view.getUint32(mdhd.start + (version === 1 ? 16 : 8));
    const stbl = find(view, trak, 'mdia', 'minf', 'stbl');
    const stsd = fullBox(find(view, stbl, 'stsd'));
    const [entry] = boxes(view, stsd.start + 4, stsd.end);
    if (!AVC_ENTRIES.has(entry?.type)) return null;
    const width = view.getUint16(entry.start + 24), height = view.getUint16(entry.start + 26);
    const config = boxes(view, entry.start + VISUAL_SAMPLE_ENTRY_BYTES, entry.end).find(b => b.type === 'avcC');
    if (!config) return null;
    const description = bytes.slice(config.start, config.end);
    return { codec: avcCodec(description), description, width, height, timescale,
      samples: samples(view, stbl, editShift(view, find(view, trak, 'edts', 'elst'))) };
  }
  return null;
}

/** The media time the edit list starts presentation at (0 without one). */
function editShift(view, elst) {
  if (!elst) return 0;
  const payload = fullBox(elst);
  if (view.getUint32(payload.start) === 0) return 0;
  const mediaTime = view.getUint8(elst.start) === 1 ? Number(view.getBigInt64(payload.start + 12)) : view.getInt32(payload.start + 8);
  return mediaTime === EMPTY_EDIT ? 0 : mediaTime;
}

/** Sample offsets, sizes, decode-order presentation times (minus `shift`) and key flags from the sample tables. */
function samples(view, stbl, shift) {
  const box = type => find(view, stbl, type);
  const durations = table(view, box('stts'), 8, at => [view.getUint32(at), view.getUint32(at + 4)]);
  const ctts = box('ctts') ? table(view, box('ctts'), 8, at => [view.getUint32(at), view.getInt32(at + 4)]) : [];
  const stsz = fullBox(box('stsz'));
  const fixedSize = view.getUint32(stsz.start), count = view.getUint32(stsz.start + 4);
  const sizes = Array.from({ length: count }, (_, i) => fixedSize || view.getUint32(stsz.start + 8 + 4 * i));
  const chunks = box('co64') ? table(view, box('co64'), 8, at => Number(view.getBigUint64(at)))
    : table(view, box('stco'), 4, at => view.getUint32(at));
  const runs = table(view, box('stsc'), 12, at => [view.getUint32(at) - 1, view.getUint32(at + 4)]);
  const sync = box('stss') ? new Set(table(view, box('stss'), 4, at => view.getUint32(at) - 1)) : null;
  const out = [];
  let sample = 0, dts = 0, durationRun = 0, durationLeft = durations[0]?.[0] ?? 0, cttsRun = 0, cttsLeft = ctts[0]?.[0] ?? 0;
  for (let chunk = 0; chunk < chunks.length; chunk++) {
    const run = runs.findLast(([first]) => first <= chunk);
    let offset = chunks[chunk];
    for (let i = 0; i < run[1] && sample < count; i++, sample++) {
      while (durationLeft === 0) durationLeft = durations[++durationRun][0];
      const duration = durations[durationRun][1];
      let composition = 0;
      if (ctts.length) { while (cttsLeft === 0) cttsLeft = ctts[++cttsRun][0]; composition = ctts[cttsRun][1]; cttsLeft--; }
      out.push({ offset, size: sizes[sample], pts: dts + composition - shift, duration, key: sync ? sync.has(sample) : true });
      offset += sizes[sample]; dts += duration; durationLeft--;
    }
  }
  return out;
}
