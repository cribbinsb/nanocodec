// Exercise the production decode worker and renderer with golden deep frames.
import { compiledModule } from '../wasm.js';
import { createRenderer } from '../renderer.js';
import { streamPackets } from '/test-packets.mjs';

const sha256 = async bytes => Array.from(new Uint8Array(
  await crypto.subtle.digest('SHA-256', bytes),
)).map(value => value.toString(16).padStart(2, '0')).join('');

export async function play({ fixture, preferred }) {
  const canvas = document.createElement('canvas');
  canvas.width = fixture.width; canvas.height = fixture.height;
  document.body.replaceChildren(canvas);
  const renderer = createRenderer(canvas, { preferred });
  if (renderer.kind !== preferred) throw Error(`renderer ${renderer.kind} instead of ${preferred}`);
  const response = await fetch(`/test-streams/${fixture.file}`);
  if (!response.ok) throw Error(`fixture HTTP ${response.status}`);
  const packets = streamPackets(new Uint8Array(await response.arrayBuffer()), fixture.codec);
  const worker = new Worker('/decode-worker.js', { type: 'module' });
  const channel = new MessageChannel();
  const next = () => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('worker timeout')), 60000);
    worker.onmessage = ({ data }) => {
      if (data.type === 'error') { clearTimeout(timer); reject(Error(data.message)); }
      else if (data.type === 'ready' || data.type === 'frame') { clearTimeout(timer); resolve(data); }
    };
    worker.onerror = event => { clearTimeout(timer); reject(Error(event.message)); };
  });
  let shown = 0;
  try {
    const ready = next();
    worker.postMessage({
      type: 'init', codec: fixture.codec, wasmModule: await compiledModule(),
      port: channel.port2, warmUp: false,
    }, [channel.port2]);
    await ready;
    for (let index = 0; index < packets.length; index++) {
      const packet = packets[index].slice();
      const reply = next();
      channel.port1.postMessage({
        id: index,
        coded: [{ packet, key: index % 30 === 0, timestamp: index / 30, colour: 'bt709' }],
        display: renderer.format,
      }, [packet.buffer]);
      const frame = await reply;
      if (frame.held) continue;
      if (frame.width !== fixture.width || frame.height !== fixture.height) throw Error('display dimensions');
      const expected = fixture.frames[shown][renderer.format === 'i420' ? 'i420_sha256' : 'rgba_sha256'];
      if (await sha256(frame.pixels) !== expected) throw Error(`${fixture.codec} ${preferred} pixels ${shown}`);
      const released = renderer.show(frame);
      if (released) worker.postMessage({ type: 'recycle', buffer: released }, [released]);
      shown++;
    }
    if (shown !== fixture.frames.length) throw Error(`shown frame count ${shown}`);
    let pixels;
    if (preferred === 'cpu') {
      pixels = renderer.context.getImageData(0, 0, fixture.width, fixture.height).data;
    } else {
      const gl = renderer.gl;
      const raw = new Uint8Array(fixture.width * fixture.height * 4);
      gl.readPixels(0, 0, fixture.width, fixture.height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
      pixels = new Uint8Array(raw.length);
      const row = fixture.width * 4;
      // readPixels starts at the bottom of the framebuffer; the CPU output is top-down.
      for (let y = 0; y < fixture.height; y++) {
        pixels.set(raw.subarray((fixture.height - 1 - y) * row, (fixture.height - y) * row), y * row);
      }
    }
    const canvasHash = await sha256(pixels);
    if (canvasHash !== fixture.frames.at(-1).rgba_sha256) throw Error('rendered canvas differs from RGBA');
    return { codec: fixture.codec, depth: fixture.depth, renderer: renderer.kind, frames: shown, canvas_sha256: canvasHash };
  } finally {
    worker.terminate(); channel.port1.close();
  }
}
