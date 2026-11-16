// Deep streams through the real browser binding: exact native words plus
// downshifted I420 and RGBA display. Fixtures/goldens are external, not raw YUV.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {streamPackets} from '../../tools/web/packets.mjs';
const [pkg,streams,goldenFile,reportFile,eightFile]=process.argv.slice(2);
if(!goldenFile) throw new Error('usage: high-depth.mjs PKG STREAMS GOLDEN_JSON [REPORT_JSON] [EIGHT_BIT_MANIFEST]');
const module=await import(pathToFileURL(path.resolve(pkg,'nc_web.js')));
const wasm=await module.default({module_or_path:await readFile(path.join(pkg,'nc_web_bg.wasm'))});
const golden=JSON.parse(await readFile(goldenFile));
const md5=bytes=>createHash('md5').update(bytes).digest('hex');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const eight=eightFile?JSON.parse(await readFile(eightFile)):[];
const report=[];
for(const fixture of golden.streams){
  const packets=streamPackets(new Uint8Array(await readFile(path.join(streams,fixture.file))),fixture.codec);
  const decoder=new module.BrowserDecoder(fixture.codec);
  const frames=[];
  try{
    assert.equal(decoder.bit_depth,0);
    for(const packet of packets){
      decoder.decode(packet);
      if(!decoder.frame_shown)continue;
      const index=frames.length;
      assert.equal(decoder.bit_depth,fixture.depth);
      assert.equal(decoder.chroma_bit_depth,fixture.depth);
      assert.equal(decoder.width,fixture.width);assert.equal(decoder.height,fixture.height);
      const native=decoder.decoded_i420_u16();
      assert.equal(md5(new Uint8Array(native.buffer,native.byteOffset,native.byteLength)),fixture.frames[index].native_md5,`${fixture.codec} native ${index}`);
      const display=decoder.decoded_i420();
      assert.equal(md5(display),fixture.frames[index].display_md5,`${fixture.codec} display ${index}`);
      const offset=decoder.i420_frame();
      assert.equal(md5(new Uint8Array(wasm.memory.buffer,offset,display.length)),fixture.frames[index].display_md5);
      const rgbaOffset=decoder.rgba_frame('bt709');
      const rgba=new Uint8Array(wasm.memory.buffer,rgbaOffset,fixture.width*fixture.height*4);
      for(let alpha=3;alpha<rgba.length;alpha+=4)assert.equal(rgba[alpha],255);
      frames.push({i420_sha256:sha(display),rgba_sha256:sha(rgba)});
    }
    assert.equal(frames.length,fixture.frames.length,`${fixture.codec} shown frames`);
    const low=eight.find(entry=>entry.codec===fixture.codec);
    if(low){
      const lowPackets=streamPackets(new Uint8Array(await readFile(low.file)),fixture.codec);
      decoder.decode(lowPackets[0]);
      assert.equal(decoder.bit_depth,8,`${fixture.codec} deep to eight`);
      const words=decoder.decoded_i420_u16(),bytes=decoder.decoded_i420();
      assert.equal(words.length,bytes.length);
      assert.ok(words.every((value,index)=>value===bytes[index]),'eight-bit native words');
      decoder.decode(packets[0]);
      assert.equal(decoder.bit_depth,fixture.depth,`${fixture.codec} eight to deep`);
      const restored=decoder.decoded_i420_u16();
      assert.equal(md5(new Uint8Array(restored.buffer,restored.byteOffset,restored.byteLength)),fixture.frames[0].native_md5);
    }
  }finally{decoder.free();}
  report.push({...fixture,frames});
}
const result={status:'pass',streams:report};
if(reportFile)await writeFile(reportFile,JSON.stringify(result));
console.log(JSON.stringify({status:'pass',streams:report.map(s=>({codec:s.codec,depth:s.depth,frames:s.frames.length}))}));
