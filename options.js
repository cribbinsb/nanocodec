// Encoder options for a panel's Advanced section, read from the codec module
// itself (nc-web `options.rs`), so the page never keeps its own copy of the
// engine's parameter registry: what each codec supports, its parameters
// (name, range, default, description) and the check of free-form overrides.
// The module loads on the page only once an Advanced section is first opened.
import { MODULE_URL, compiledModule } from './wasm.js';
import { streamExtension } from './codecs.js';

let loading = null;

/**
 * The codec module's option queries, once it has loaded (later calls resolve at once):
 * - `codecOptions(codec)`: `{ filmGrain, blockQp, temporalLayers, secondReference }`;
 * - `parameterRegistry(codec)`: `[{ name, default, min, max, doc }]`, defaults being the codec's;
 * - `checkParameters(codec, text)`: the bad lines of `name=value` text, `[{ line, message }]`.
 * They are synchronous, so a codec change can re-check the settings before the restart reads them.
 */
export function loadOptions() {
  loading ??= import(MODULE_URL).then(async module => {
    await module.default({ module_or_path: await compiledModule() });
    const registries = new Map();
    return {
      codecOptions: codec => JSON.parse(module.codec_options(codec)),
      parameterRegistry: codec => {
        if (!registries.has(codec)) registries.set(codec, JSON.parse(module.parameter_registry(codec)));
        return registries.get(codec);
      },
      checkParameters: (codec, text) => JSON.parse(module.check_parameters(codec, text)),
    };
  });
  return loading;
}

// What the curated toggles stand for, as parameter settings.
// Persistence AQ: the probed setting of docs/handoff/block-aq.md (persist 2 + still 3).
export const PERSISTENCE_AQ = ['persist.strength=2'];
export const SECOND_REFERENCE = ['refs.second=1'];
export const layerCount = count => (count > 1 ? [`layers.count=${count}`] : []);

/** The CLI's `--preset` for the page's preset choice, or null for Default (nc-web `preset`). */
export function cliPreset(codec, preset) {
  if (preset === 'fast') return `${codec}-fast`;
  if (preset === 'faster') return `${codec}-faster`;
  if (preset === 'fastest') return `${codec}-fastest`;
  if (preset === 'svc') return `${codec}-svc`;
  if (preset === 'svc-fast') return `${codec}-svc-fast`;
  if (preset === 'svc3') return `${codec}-svc3`;
  if (preset === 'svc3-fast') return `${codec}-svc3-fast`;
  return null;
}

/** The settings lines of an override text: comments and blank lines dropped. */
export function settingLines(text) {
  return text.split('\n').map(line => line.split('#')[0].trim()).filter(Boolean)
    .map(line => line.split('=').map(part => part.trim()).join('='));
}

/**
 * The `nanocodec encode` command with the settings of `config` (as Panel.config returns it)
 * for `frames` frames: codec, size, rate, key interval, quantizer seed, preset, overrides.
 */
export function cliCommand(config, frames) {
  const args = ['nanocodec encode', `--codec ${config.codec}`, '--input source.yuv',
    `--width ${config.width}`, `--height ${config.height}`, `--fps ${config.fps}`, `--frames ${frames}`,
    `--intra-period ${config.keyInterval}`, `--qp ${config.qp}`];
  if (config.rateMode === 'cbr') args.push(`--bitrate ${config.targetKbps}`, `--peak-bitrate ${config.peakKbps}`, `--buffer-ms ${config.bufferMs}`);
  const preset = cliPreset(config.codec, config.preset);
  if (preset) args.push(`--preset ${preset}`);
  for (const setting of settingLines(config.overrides)) args.push(`--param ${setting}`);
  if (config.filmGrain) args.push('--film-grain');
  args.push(`--output stream.${streamExtension(config.codec)}`);
  return args.join(' ');
}
