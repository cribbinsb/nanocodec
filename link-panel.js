// A live view's encoder-to-decoder link (rtp-link.js): the controls that pick
// it (Bitstream, RTP, or RTP over the in-page lossy network), the network and
// resilience settings, and, while it runs, the live statistics strip and the
// event timeline (loss -> feedback -> recovery). Panel (panel.js) owns one;
// what each codec supports comes from the codec module (options.js
// `codecOptions`, the engine's own answer), so nothing here keeps a copy of
// the support matrix of docs/FAQ.md, only the reasons a control is off.
import { loadOptions } from './options.js';
import { DEFAULT_NETWORK, DEFAULT_PRESET, LOSS_SCOPES, NETWORK_PRESETS, roundTripMs } from './network.js';
import { TIMELINE_LANES } from './rtp-link.js';

export const LINK_MODES = { bitstream: 'bitstream', rtp: 'rtp', lossy: 'lossy' };
/** Frames between long-term reference marks with LTR feedback on: the loss simulator's checked setting (docs/handoff/loss-resilience.md). */
const LTR_INTERVAL = 15;
/**
 * The packet loss rate (0..1) loss-aware CBR takes as its hint: the lossy network's
 * loss setting while `link` runs over it with loss-aware CBR on, else null.
 */
export function lossRateHint(link) {
  if (link?.mode !== LINK_MODES.lossy || !link.resilience?.lossAwareRc) return null;
  return (link.network?.lossPercent ?? 0) / 100;
}
/** The intra-refresh sweep: one second of frames. */
const refreshPeriod = fps => fps;
/** The timeline shows this much source time. */
export const TIMELINE_MS = 10000;
const CUSTOM = 'custom';
/** Over RTP, temporal layers keep their structure on every frame (see TIPS.temporalLayers). */
const FIXED_LAYERS = 'layers.adaptive=0';

const TIPS = {
  bitstream: 'The encoder\'s packets go straight to the decoder: nothing is lost or delayed.',
  rtp: 'The encoder\'s output is packetized as RTP by nanocodec-rtp (RFC 6184 / 7798 / 9328, the AV1 and VP9 payload formats, with the Dependency Descriptor), sent over a perfect network, depacketized and decoded.',
  lossy: 'RTP over an emulated network in this page: loss, delay, jitter, reordering and a bandwidth cap, with the receiver\'s loss feedback back to the encoder one round trip later. The picture is what the decoder really outputs: a lost frame freezes it until a repair arrives (no concealment).',
  noRtp: 'AV2 has no RTP payload format yet (docs/handoff/rtp.md)',
  vp9Spatial: 'VP9 spatial layers are not carried over RTP: our VP9 SVC shows lower layers with show_existing_frame, which RFC 9628\'s picture model does not carry (docs/handoff/rtp.md). Temporal layers are.',
  ltr: `Long-term reference with loss feedback: the encoder marks a long-term reference every ${LTR_INTERVAL} frames; the receiver acknowledges the frames it decodes and reports the ones it lost, and the encoder repairs from the newest acknowledged long-term reference instead of a key frame (ltr.interval=${LTR_INTERVAL}).`,
  ltrSpatial: 'Long-term references are not combined with spatial layers. With spatial layers the receiver reports each lost layer picture anyway: a lost enhancement layer is repaired from the layer below (the base layer is shown meanwhile), a lost base layer by a key frame.',
  refresh: 'Intra refresh: a column of intra blocks sweeps the picture every second, so there are no large key frames. Our decoders refuse a frame whose reference was lost, so after a loss the repair still comes from the long-term reference or a key frame (docs/FAQ.md).',
  refreshUnsupported: 'Refused for this codec: VP9\'s backward-adapted probabilities and previous-frame motion vectors, and AV2\'s primary_ref_frame, carry state a lost frame changes, so a sweep does not recover a standard decoder (docs/FAQ.md)',
  refreshSpatial: 'Intra refresh is not combined with spatial layers (docs/FAQ.md)',
  refreshSecondRef: 'Intra refresh is not combined with the second reference (Advanced)',
  keyRequests: 'Key frame on request: when the receiver cannot decode (a loss with no long-term reference to repair from), it asks the encoder for a key frame (a PLI), repeated every round trip while the picture stays frozen. Off: the picture stays frozen until the next periodic key frame.',
  lossAwareRc: 'Loss-aware CBR (rate.loss_aware=1, CBR only; off by default): the encoder estimates packet loss from the receiver\'s feedback (and, over the lossy network, takes its loss setting as a hint, live), caps key and large inter frames so they are likely to arrive whole, shrinks key frames further after each lost key, and spends the bits a capped key saved on the frames after it (crates/nc-encoder/src/loss_rate.rs).',
  lossAwareSpatial: 'Loss-aware CBR is not combined with spatial layers: each layer\'s rate control codes without the receiver\'s feedback',
  temporalLayers: 'Temporal layers: no frame predicts from the top layer, so a lost top-layer frame is dropped by the receiver and the picture goes on (a skip, not a freeze). Over RTP the structure is fixed (layers.adaptive=0): the presets\' adaptive layers would code flat while the content allows.',
};

const escapeHtml = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const options = (items, selected) => items.map(([value, label]) => `<option value="${value}"${String(value) === String(selected) ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('');
const numberField = (name, label, value, min, max, step, unit) =>
  `<label class="compact-control">${label}<span class="unit-input"><input class="${name}" type="number" min="${min}" max="${max}" step="${step}" value="${value}"><small>${unit}</small></span></label>`;
/** Disable `control` (kept in place, dimmed, `why` as its tooltip) or enable it with its own tooltip. */
function offer(box, control, on, tip, why) {
  control.disabled = !on;
  if (!on && control.type === 'checkbox') control.checked = false;
  box.classList.toggle('inapplicable', !on);
  box.title = on ? tip : why;
}

export class LinkControls {
  /** `changed(kind)`: 'network' for settings the running link takes live, else a restart. `temporalLayers`: Advanced's layer control (moved here). */
  constructor(letter, changed, temporalLayers) {
    this.changed = changed; this.supported = null; this.codec = null;
    this.element = document.createElement('div'); this.element.className = 'link-options live-only';
    const n = DEFAULT_NETWORK;
    this.element.innerHTML = `
      <div class="link-row"><span class="control-label">LINK</span><div class="segmented link-mode" role="group" aria-label="Version ${letter}: how packets reach the decoder">
        <button type="button" data-link="${LINK_MODES.bitstream}" class="selected" aria-pressed="true" title="${escapeHtml(TIPS.bitstream)}">Bitstream</button><button type="button" data-link="${LINK_MODES.rtp}" aria-pressed="false" title="${escapeHtml(TIPS.rtp)}">RTP</button><button type="button" data-link="${LINK_MODES.lossy}" aria-pressed="false" title="${escapeHtml(TIPS.lossy)}">RTP · lossy network</button></div></div>
      <div class="link-settings" hidden>
        <label class="compact-control network-only">Network<select class="network-preset">${options([...Object.entries(NETWORK_PRESETS).map(([id, p]) => [id, p.label]), [CUSTOM, 'Custom (Network emulator below)']], DEFAULT_PRESET)}</select></label>
        <div class="resilience"><span class="control-label">RESILIENCE</span>
          <label class="toggle ltr-option"><input class="ltr" type="checkbox" checked>Long-term reference + loss feedback</label>
          <label class="toggle refresh-option"><input class="refresh" type="checkbox">Intra refresh</label>
          <label class="toggle key-option" title="${escapeHtml(TIPS.keyRequests)}"><input class="key-requests" type="checkbox" checked>Key frame on request</label>
          <label class="toggle loss-aware-option" title="${escapeHtml(TIPS.lossAwareRc)}"><input class="loss-aware" type="checkbox">Loss-aware CBR</label>
        </div>
      </div>
      <details class="network-advanced" hidden><summary>Network emulator</summary><div class="network-grid">
        <label class="compact-control">Loss model<select class="loss-model">${options([['random', 'Random (independent)'], ['burst', 'Bursts (Gilbert-Elliott)']], n.lossModel)}</select></label>
        ${numberField('loss-percent', 'Loss', n.lossPercent, 0, 50, 0.5, '%')}
        ${numberField('burst-length', 'Mean burst', n.burstLength, 1, 20, 1, 'packets')}
        <label class="compact-control">Loss applies to<select class="loss-scope">${options(Object.entries(LOSS_SCOPES), n.lossScope)}</select></label>
        ${numberField('delay', 'Delay (one way)', n.delayMs, 0, 1000, 5, 'ms')}
        ${numberField('jitter', 'Jitter', n.jitterMs, 0, 500, 1, 'ms')}
        ${numberField('reorder', 'Reorder', n.reorderPercent, 0, 50, 0.5, '%')}
        ${numberField('duplicate', 'Duplicate', n.duplicatePercent, 0, 50, 0.5, '%')}
        ${numberField('bandwidth', 'Bandwidth cap (0: none)', n.bandwidthKbps, 0, 100000, 50, 'kbps')}
        ${numberField('queue', 'Queue', n.queueMs, 10, 5000, 10, 'ms')}
        ${numberField('mtu', 'MTU', n.mtu, 200, 1500, 50, 'bytes')}
        ${numberField('rtt', 'Feedback RTT (empty: 2 × delay)', '', 0, 5000, 5, 'ms')}
        ${numberField('seed', 'Loss seed', n.seed, 1, 999999, 1, '')}
        <label class="compact-control sfu-control">SFU forwards spatial<select class="sfu-spatial"></select></label>
        <label class="compact-control sfu-control">SFU forwards temporal<select class="sfu-temporal"></select></label>
      </div><p class="network-note">Delay, jitter and loss apply live; MTU, the SFU and the resilience toggles restart the stream. The SFU drops layers by the Dependency Descriptor alone, before the network, as a selective forwarding server does.</p></details>`;
    const $ = selector => this.element.querySelector(selector);
    this.modeButtons = [...this.element.querySelectorAll('[data-link]')];
    this.mode = LINK_MODES.bitstream;
    this.settingsBox = $('.link-settings'); this.advanced = $('.network-advanced');
    this.preset = $('.network-preset');
    this.ltr = $('.ltr'); this.refresh = $('.refresh'); this.keyRequests = $('.key-requests'); this.lossAware = $('.loss-aware');
    this.temporalLayers = temporalLayers;
    temporalLayers.title = TIPS.temporalLayers;
    $('.resilience').append(temporalLayers);
    this.fields = {
      lossModel: $('.loss-model'), lossPercent: $('.loss-percent'), burstLength: $('.burst-length'), lossScope: $('.loss-scope'),
      delayMs: $('.delay'), jitterMs: $('.jitter'), reorderPercent: $('.reorder'), duplicatePercent: $('.duplicate'),
      bandwidthKbps: $('.bandwidth'), queueMs: $('.queue'), rttMs: $('.rtt'), seed: $('.seed'),
    };
    this.mtu = $('.mtu'); this.sfuSpatial = $('.sfu-spatial'); this.sfuTemporal = $('.sfu-temporal');
    for (const button of this.modeButtons) button.addEventListener('click', () => {
      if (button.dataset.link === this.mode || button.disabled) return;
      this.setMode(button.dataset.link); this.modeChanged?.(); changed();
    });
    this.preset.addEventListener('change', () => { this.applyPreset(); changed('network'); });
    for (const field of Object.values(this.fields)) field.addEventListener('change', () => {
      if (!field.checkValidity()) { field.reportValidity(); return; }
      if (Object.keys(NETWORK_PRESETS.good).includes(this.fieldName(field))) this.preset.value = CUSTOM;
      changed('network');
    });
    for (const control of [this.ltr, this.refresh, this.keyRequests, this.lossAware, this.mtu, this.sfuSpatial, this.sfuTemporal]) control.addEventListener('change', () => changed());
    // The visitor's LTR choice, restored when spatial layers no longer rule it out.
    this.ltr.addEventListener('change', () => { this.ltrWanted = this.ltr.checked; });
  }
  fieldName(field) { return Object.entries(this.fields).find(([, f]) => f === field)[0]; }
  /** Choose the link mode and show its settings. */
  setMode(mode) {
    this.mode = mode;
    for (const button of this.modeButtons) {
      const selected = button.dataset.link === mode;
      button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected));
    }
    this.settingsBox.hidden = mode === LINK_MODES.bitstream;
    this.advanced.hidden = mode !== LINK_MODES.lossy;
    for (const element of this.element.querySelectorAll('.network-only')) element.hidden = mode !== LINK_MODES.lossy;
  }
  /** Put the preset's loss, burst, delay and jitter into the fields. */
  applyPreset() {
    const preset = NETWORK_PRESETS[this.preset.value];
    if (!preset) return;
    for (const key of ['lossPercent', 'burstLength', 'delayMs', 'jitterMs']) this.fields[key].value = String(preset[key]);
    this.fields.lossModel.value = preset.burstLength > 1 ? 'burst' : 'random';
  }
  /**
   * The view now codes `codec` with `spatialLayers` spatial layers, `temporalLayers`
   * temporal layers and the second reference on or off: offer what applies, with the
   * reason on what does not. Loads the codec module's options once.
   */
  async update(codec, { spatialLayers, temporalLayers, secondReference }) {
    this.codec = codec;
    this.layers = { spatialLayers, temporalLayers };
    try { this.supported = (await loadOptions()).codecOptions(codec); } catch { this.supported = null; }
    if (this.codec !== codec) return;
    const s = this.supported ?? {};
    const rtpOff = !s.rtp ? TIPS.noRtp : codec === 'vp9' && spatialLayers > 1 ? TIPS.vp9Spatial : null;
    for (const button of this.modeButtons.slice(1)) {
      button.disabled = Boolean(rtpOff);
      button.title = rtpOff ?? TIPS[button.dataset.link];
    }
    if (rtpOff && this.mode !== LINK_MODES.bitstream) { this.setMode(LINK_MODES.bitstream); this.changed(); }
    const ltrWhy = !s.longTermReference ? 'This codec has no long-term references' : spatialLayers > 1 ? TIPS.ltrSpatial : null;
    offer(this.element.querySelector('.ltr-option'), this.ltr, !ltrWhy, TIPS.ltr, ltrWhy);
    if (!ltrWhy && this.ltrWanted !== undefined) this.ltr.checked = this.ltrWanted;
    const refreshWhy = !s.intraRefresh ? TIPS.refreshUnsupported : spatialLayers > 1 ? TIPS.refreshSpatial : secondReference ? TIPS.refreshSecondRef : null;
    offer(this.element.querySelector('.refresh-option'), this.refresh, !refreshWhy, TIPS.refresh, refreshWhy);
    offer(this.element.querySelector('.loss-aware-option'), this.lossAware, spatialLayers <= 1, TIPS.lossAwareRc, TIPS.lossAwareSpatial);
    const fill = (select, count, name) => {
      // "All layers" stays all layers when the count changes; a chosen limit is kept where it still exists.
      const all = select.value === '' || +select.value === select.options.length - 1;
      const value = all ? count - 1 : Math.min(+select.value, count - 1);
      select.innerHTML = options(Array.from({ length: count }, (_, i) => [i, i === count - 1 ? `all (${name}${i})` : `up to ${name}${i}`]), value);
      select.disabled = count < 2;
      select.closest('label').title = count < 2 ? `The stream has one ${name === 'S' ? 'spatial' : 'temporal'} layer` : '';
    };
    fill(this.sfuSpatial, spatialLayers, 'S'); fill(this.sfuTemporal, temporalLayers, 'T');
  }
  /** The network settings now in the fields. */
  network() {
    const value = key => +this.fields[key].value;
    return { ...DEFAULT_NETWORK, lossModel: this.fields.lossModel.value, lossScope: this.fields.lossScope.value,
      lossPercent: value('lossPercent'), burstLength: value('burstLength'), delayMs: value('delayMs'), jitterMs: value('jitterMs'),
      reorderPercent: value('reorderPercent'), duplicatePercent: value('duplicatePercent'), bandwidthKbps: value('bandwidthKbps'),
      queueMs: value('queueMs'), rttMs: this.fields.rttMs.value === '' ? null : value('rttMs'), seed: value('seed'), mtu: +this.mtu.value };
  }
  /**
   * The link settings a view runs with: `{ mode, network, resilience, sfu }`, and the
   * encoder parameter lines the resilience toggles stand for (`overrides`).
   */
  settings(fps, temporalLayers) {
    if (this.mode === LINK_MODES.bitstream) return { link: { mode: this.mode }, overrides: [] };
    const network = this.network();
    const resilience = { ltr: this.ltr.checked, intraRefresh: this.refresh.checked, keyRequests: this.keyRequests.checked, lossAwareRc: this.lossAware.checked };
    const sfu = { spatial: +(this.sfuSpatial.value || 0), temporal: +(this.sfuTemporal.value || 0) };
    const link = { mode: this.mode, network, resilience, sfu };
    // The starting hint is a parameter too, so the CLI command shown under the view codes the same stream;
    // later network changes reach the running encoder as live hints (pipeline.js `updateNetwork`).
    const hint = lossRateHint(link);
    const overrides = [...(resilience.ltr ? [`ltr.interval=${LTR_INTERVAL}`] : []), ...(resilience.intraRefresh ? [`refresh.period=${refreshPeriod(fps)}`] : []),
      ...(resilience.lossAwareRc ? ['rate.loss_aware=1', ...(hint ? [`rate.loss_rate=${hint.toFixed(3)}`] : [])] : []),
      ...(temporalLayers > 1 ? [FIXED_LAYERS] : [])];
    return { link, overrides };
  }
  /** One line for the rate status: the link and its settings. */
  static describe(link) {
    if (!link || link.mode === LINK_MODES.bitstream) return '';
    // Loss-aware CBR is a rate-control mode, not a repair: its state always shows, on or off.
    const lossAware = ` · loss-aware CBR ${link.resilience.lossAwareRc ? 'on' : 'off'}`;
    if (link.mode === LINK_MODES.rtp) return ` · RTP (MTU ${link.network.mtu})${lossAware}`;
    const on = Object.entries({ LTR: link.resilience.ltr, 'intra refresh': link.resilience.intraRefresh, 'key requests': link.resilience.keyRequests })
      .filter(([, value]) => value).map(([name]) => name);
    const n = link.network;
    return ` · RTP, ${n.lossPercent} % ${n.lossModel === 'burst' ? `bursts of ${n.burstLength}` : 'random'} loss${n.lossScope === 'all' ? '' : ` (${LOSS_SCOPES[n.lossScope].toLowerCase()})`}, ${n.delayMs} ms ± ${n.jitterMs} ms, RTT ${roundTripMs(n)} ms · ${on.length ? on.join(', ') : 'no resilience'}${lossAware}`;
  }
}

/** The live statistics strip and the event timeline under a view's picture. */
export class LinkMonitor {
  constructor() {
    this.element = document.createElement('div'); this.element.className = 'link-monitor'; this.element.hidden = true;
    this.element.innerHTML = `<div class="link-stats" role="status" aria-live="off"></div>
      <div class="link-timeline"><div class="history-label"><span>EVENTS · LAST ${TIMELINE_MS / 1000} S</span><span class="timeline-legend">${
        Object.values(TIMELINE_LANES).map(({ label, colour }) => `<i style="--c:${colour}"></i>${label}`).join('')}</span></div><canvas width="540" height="64" aria-label="Loss, feedback and recovery events"></canvas></div>`;
    this.stats = this.element.querySelector('.link-stats'); this.canvas = this.element.querySelector('canvas');
  }
  reset(link) {
    this.element.hidden = !link || link.mode === LINK_MODES.bitstream;
    this.events = []; this.now = 0; this.state = null;
    this.stats.innerHTML = '';
    this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
  /** A decode result's link state (rtp-link.js `RtpLink.state`): keep its new events. */
  update(state) {
    if (!state) return;
    this.state = state; this.now = state.now;
    this.events.push(...state.events);
    this.events = this.events.filter(e => e.at >= this.now - TIMELINE_MS);
  }
  /** Redraw the strip and the timeline (as often as the panel's metrics). */
  draw() {
    const s = this.state;
    if (!s || this.element.hidden) return;
    const chip = (label, value, title, warn = false) => `<span class="link-stat${warn ? ' warn' : ''}" title="${escapeHtml(title)}"><small>${label}</small><b>${value}</b></span>`;
    const recover = s.recoveries.length ? `${Math.round(s.recoveries.at(-1))} ms` : '—';
    const meanRecover = s.recoveries.length ? `mean ${Math.round(s.recoveries.reduce((a, b) => a + b, 0) / s.recoveries.length)} ms over ${s.recoveries.length}` : 'no freeze yet';
    const layers = Object.entries(s.layerLoss).map(([layer, { sent, lost }]) => `${layer} ${lost}/${sent}`).join(' · ');
    this.stats.innerHTML = [
      chip('Packets', `${s.packetsSent}`, 'RTP packets sent by the encoder\'s side'),
      chip('Lost', `${s.packetsLost}${s.packetsSent ? ` (${(100 * s.packetsLost / s.packetsSent).toFixed(1)} %)` : ''}`, `Lost by the network or its queue, per layer (lost/sent): ${layers || 'none'}${s.sfuDropped ? `; ${s.sfuDropped} more dropped by the SFU on purpose` : ''}`, s.packetsLost > 0),
      chip('Late', `${s.packetsLate}`, 'Packets that arrived after the jitter buffer had given them up'),
      chip('Decoded', `${s.framesDecoded}`, 'Frames the decoder decoded'),
      chip('Frozen', `${s.framesFrozen}`, 'Frames for which the picture stayed frozen: the decoder had nothing it could decode (no concealment)', s.frozen),
      chip('Dropped', `${s.framesDropped}`, 'Frames the receiver did not hand to the decoder: lost, or undecodable because a frame they predict from was lost. A dropped top-temporal-layer frame only skips; frozen frames are counted beside'),
      chip('Recovery', recover, `Time from the first frozen frame to the next decoded picture, last freeze; ${meanRecover}`),
      chip('Key requests', `${s.keyRequests}`, 'Key frame requests (PLI) the receiver sent'),
      chip('Loss reports', `${s.lossReports}`, 'Lost frames the receiver reported to the encoder'),
      chip('Wire', `${Math.round(s.wireKbps)} kbps`, 'Bitrate on the wire over the last second, with RTP, UDP and IP headers'),
      chip('On screen', s.frozen ? 'frozen' : s.shownLayer, 'Which layer the picture on screen is decoded from'),
    ].join('');
    this.drawTimeline();
  }
  drawTimeline() {
    const ctx = this.canvas.getContext('2d'), { width, height } = this.canvas;
    ctx.clearRect(0, 0, width, height);
    const lanes = Object.entries(TIMELINE_LANES), laneHeight = height / lanes.length, start = this.now - TIMELINE_MS;
    const x = at => (at - start) / TIMELINE_MS * width;
    lanes.forEach(([kind, { colour }], i) => {
      ctx.fillStyle = '#f0f2ed'; ctx.fillRect(0, i * laneHeight + laneHeight / 2 - 0.5, width, 1);
      ctx.fillStyle = colour;
      for (const event of this.events) {
        if (event.lane !== kind) continue;
        const end = event.until ?? event.at;
        ctx.fillRect(x(event.at), i * laneHeight + 3, Math.max(2, x(end) - x(event.at)), laneHeight - 6);
      }
    });
    // A loss report's way back to the encoder: from the receiver's report to its arrival.
    ctx.strokeStyle = TIMELINE_LANES.feedback.colour;
    for (const event of this.events) {
      if (event.lane !== 'feedback' || event.arrives === undefined) continue;
      ctx.beginPath(); ctx.moveTo(x(event.at), laneHeight * 1.5); ctx.lineTo(x(event.arrives), laneHeight * 2.5); ctx.stroke();
    }
  }
}
