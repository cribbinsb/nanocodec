// Decodes one conformance stream with the browser module (BrowserDecoder) as
// tools/web/conformance-runner.mjs does under node: one packet per picture
// (packets.mjs), every shown picture's cropped planes, 8-bit streams as bytes
// and deeper ones as little-endian 16-bit samples, hashed into one MD5.
import { streamPackets } from './packets.mjs';
import { Md5 } from './md5.mjs';

const modules = new Map();

/** The browser module in directory PKG (a URL), loaded once per worker. */
function load(pkg) {
  if (!modules.has(pkg)) {
    modules.set(pkg, import(`${pkg}/nc_web.js`).then(async module => {
      await module.default({ module_or_path: `${pkg}/nc_web_bg.wasm` });
      return module;
    }));
  }
  return modules.get(pkg);
}

async function decode({ pkg, codec, url, deep }) {
  const module = await load(pkg);
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const decoder = new module.BrowserDecoder(codec);
  const whole = new Md5();
  let pictures = 0;
  const take = () => {
    const samples = deep ? decoder.decoded_i420_u16() : decoder.decoded_i420();
    whole.update(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
    pictures++;
  };
  const drain = () => { while (decoder.next_frame()) take(); };
  try {
    for (const packet of streamPackets(bytes, codec)) {
      decoder.decode(packet);
      if (decoder.frame_shown) take();
      drain();
    }
    decoder.finish();
    drain();
    return { pictures, md5: whole.hex() };
  } catch (error) {
    return { pictures, error: String(error) };
  } finally {
    decoder.free();
  }
}

self.onmessage = async e => self.postMessage(await decode(e.data));
