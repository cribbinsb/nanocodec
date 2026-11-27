// The page's CLI line (options.js cliCommand) signals the colour the browser encoder
// signals (nc-web signalled_colour), so the command reproduces the stream byte for byte
// (scene-sanity 2026-11-22: without it the H.264 SPS and AV1 sequence header differed).
import assert from 'node:assert/strict';
import test from 'node:test';
import { cliColour, cliCommand } from '../options.js';

const config = { codec: 'av1', width: 854, height: 480, fps: 30, keyInterval: 900, qp: 180, rateMode: 'cbr',
  targetKbps: 400, peakKbps: 800, bufferMs: 1000, preset: 'medium', spatialLayers: 1, overrides: '',
  filmGrain: false, stillCamera: false, keyAtClipStart: false };

test('the CLI line signals the coded colour', () => {
  assert.match(cliCommand(config, 60), / --colour bt601 --output stream\.ivf$/);
  assert.match(cliCommand(config, 60, 'bt709-full'), / --colour bt709 --range full /);
  assert.deepEqual(cliColour('h264', 'bt601'), ['--colour bt601']);
  assert.deepEqual(cliColour('av2', 'bt601'), [], 'AV2 signals no colour');
});
