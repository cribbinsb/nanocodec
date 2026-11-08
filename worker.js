// One independent codec state per comparison view. No frames queue behind encoding.
import init, { BrowserEncoder, BrowserDecoder } from './pkg/nc_web.js';
let encoder, decoder;
function configureRate(c) {
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  else encoder.disable_rate_control();
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      await init();
      const c = message.config;
      encoder = new BrowserEncoder(c.codec, c.width, c.height, c.qp, c.fps, c.keyInterval, c.fast);
      decoder = new BrowserDecoder(c.codec);
      configureRate(c);
      self.postMessage({ type: 'ready' });
    } else if (message.type === 'keyframe') {
      encoder.request_keyframe();
    } else if (message.type === 'rate') {
      configureRate(message.config);
    } else if (message.type === 'frame') {
      const start = performance.now();
      const packet = encoder.encode(new Uint8Array(message.rgba));
      const encoded = performance.now();
      const rgba = decoder.decode(packet);
      const decoded = performance.now();
      const psnr = encoder.quality(decoder);
      self.postMessage({ type: 'frame', id: message.id, timestamp: message.timestamp,
        packet, rgba, key: encoder.key, width: decoder.width, height: decoder.height,
        encodeMs: encoded - start, decodeMs: decoded - encoded, psnr, quantizer: encoder.quantizer, bufferFullness: encoder.buffer_fullness, bufferOverflows: encoder.buffer_overflows }, [packet.buffer, rgba.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
