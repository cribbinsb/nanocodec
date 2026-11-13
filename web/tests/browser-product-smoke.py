#!/usr/bin/env python3
"""Bounded real Chromium demo smoke with synthetic webcam, isolated HTTP/profile."""
import functools
import http.server
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
from playwright.sync_api import sync_playwright

web = Path(__file__).resolve().parents[1]
class Quiet(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP range requests, as real hosts serve them: a browser
    seeks a video (the page's Step) only when the server answers ranges."""
    def log_message(self, *_):
        pass

    def send_head(self):
        requested = re.fullmatch(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
        path = self.translate_path(self.path)
        if not requested or not os.path.isfile(path):
            return super().send_head()
        with open(path, 'rb') as f:
            data = f.read()
        start = int(requested[1])
        end = min(int(requested[2] or len(data) - 1), len(data) - 1)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        return io.BytesIO(data[start:end + 1])

    def copyfile(self, source, outputfile):
        # The browser drops range requests it no longer needs (a seek, a loop).
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass

server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(web)))
threading.Thread(target=server.serve_forever, daemon=True).start()
errors, notices = [], []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        # Every failure notice and session line the page shows, printed if the test fails.
        page.on('console', lambda m: notices.append(m.text) if m.text.startswith('page: ') else None)
        page.add_init_script('''addEventListener('DOMContentLoaded', () => {
          for (const id of ['notice', 'session-stats']) new MutationObserver(() =>
            console.log(`page: ${id}: ${document.getElementById(id).textContent}`)).observe(document.getElementById(id), { childList: true, characterData: true, subtree: true });
        });''')
        # A live view has an encode and a decode worker (pipeline.js), created in that
        # order; a pre-encoded view has a decode worker only; one capture worker feeds
        # every view. The hook records the decoded frames, the encoder configs, which
        # worker posts what to the page, the frames in flight per view (sent to the capture
        # worker for a view that codes the tick, not yet back from the view's decoder; a
        # stopped view's workers are terminated with frames still in them) and the most
        # frames one view has had in flight. Each
        # decoded frame records its view (a counter over every view ever created), since
        # the two views' decoders finish frames in no fixed order.
        page.add_init_script('''window.workerFrames=[];window.workerConfigs=[];window.workerScripts=[];
          window.pagePosts=new Set();window.maxInFlight=0;const views=[];
          window.framesInFlight=()=>views.filter(v=>!v.terminated).reduce((n,v)=>n+v.sent-v.received,0);
          const Original=Worker;window.Worker=class extends Original {
            constructor(url,...args){super(url,...args);const script=String(url).split('/').pop();
              window.workerScripts.push(script);
              if(script==='encode-worker.js')views.push(this.view={sent:0,received:0,decoder:false});
              else if(script==='decode-worker.js'){if(!views.length||views.at(-1).decoder)views.push({sent:0,received:0});this.view=views.at(-1);this.view.decoder=true;}
              this.addEventListener('message', e=>{
              const d=e.data;window.pagePosts.add(script+':'+d.type);if(d.held)window.heldFrames=(window.heldFrames||0)+1;
              if(d.type==='frame')this.view.received++;
              if(d.pixels)window.workerFrames.push({view:views.indexOf(this.view),key:d.key,format:d.format,width:d.width,height:d.height,packet:d.packet.length,earlier:(d.earlier||[]).length,psnr:d.psnr});
              window.repeatFrames=(window.repeatFrames||0)+Number(!!d.repeat)+(d.earlier||[]).filter(e=>e.repeat).length;
            });}
            postMessage(d,...rest){if(d.config)window.workerConfigs.push(d.config);
              if(d.type==='frame')views.filter(v=>!v.terminated).forEach((v,i)=>{if(d.coding[i]){v.sent++;window.maxInFlight=Math.max(window.maxInFlight,v.sent-v.received);}});
              super.postMessage(d,...rest);}
            terminate(){if(this.view)this.view.terminated=true;super.terminate();}
          };''')
        # Headless Chromium has only software GL (SwiftShader), which the automatic
        # choice refuses: ask for WebGL2 explicitly to exercise the GL display.
        page.goto(f'http://127.0.0.1:{server.server_port}/?display=webgl2')
        # The page opens already comparing the default sample clip, without a user gesture.
        # Each view has decoded two frames (the faster view can deliver several before
        # the slower one's first, as frames are in flight).
        page.wait_for_function('[0, 1].every(v => window.workerFrames.filter(f => f.view === v).length >= 2)', timeout=60000)
        assert page.locator('#clip-credit').is_visible()
        assert page.evaluate("!document.querySelector('#input-video').paused")
        # H.264 and AV1 under CBR hold frame 0 back for the start-up lookahead,
        # then deliver it with frame 1.
        assert page.evaluate('window.heldFrames') == 2
        assert page.evaluate('window.workerFrames.filter(f => f.earlier).length') == 2
        # Two workers per view and one capture worker; frames pass from capture to
        # encoder to decoder over MessageChannels, so the encode workers post nothing
        # but 'ready' and, once, the start-up warm-up's report (encode-warmup.js) to the page.
        assert sorted(page.evaluate('window.workerScripts')) == ['capture-worker.js', 'decode-worker.js', 'decode-worker.js', 'encode-worker.js', 'encode-worker.js']
        assert set(page.evaluate("[...window.pagePosts].filter(p => p.startsWith('encode-worker.js:'))")) <= {'encode-worker.js:ready', 'encode-worker.js:warmed'}
        # PSNR comes from the decode worker for every frame.
        assert page.evaluate('window.workerFrames.every(f => f.psnr > 20)')
        # The session line names the busiest worker.
        page.wait_for_function(r"/fps delivered · (keeping up, slowest|limited by) (encode|decode) [AB] \(\d+ ms\)/.test(document.querySelector('#session-stats').textContent)", timeout=10000)
        autostart_configs = page.evaluate('window.workerConfigs')
        assert all(c['targetKbps'] == 300 and c['keyInterval'] == 30 * c['fps'] for c in autostart_configs)
        # Headless Chromium on SwiftShader loses new GL contexts once and restores them ~1 s later.
        page.wait_for_function("[...document.querySelectorAll('.viewport canvas')].every(c => !c.getContext('webgl2').isContextLost())", timeout=10000)
        # Magnifier: a tap on version B pins one region; every pane (A, B, original) gets pixels.
        page.locator('.loupe-toggle').click()
        page.locator('.viewport canvas').nth(1).click(position={'x': 100, 'y': 60})
        assert page.locator('.loupe-pin').get_attribute('aria-pressed') == 'true'
        page.wait_for_timeout(300)
        assert page.evaluate("""[...document.querySelectorAll('.loupe-panes canvas')].every(c =>
          c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v))""")
        # Both views draw I420 with WebGL2 (renderer.js).
        assert page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.dataset.display)") == ['webgl2', 'webgl2']
        assert page.evaluate("window.workerFrames.every(f => f.format === 'i420')")
        page.locator('.loupe-toggle').click()
        # Clips come in 360p and 720p files (clips.js): the default 854x480 reads the 720p
        # one (scaled down), picking a clip restarts on it, and 640x360 switches to the 360p file.
        clip_src = "document.querySelector('#input-video').currentSrc"
        assert page.evaluate(clip_src).endswith('/clips/netflix-bar-scene-720p.mp4')
        assert page.evaluate('window.workerFrames.every(f => f.width === 854 && f.height === 480)')
        # The masthead links to the project page.
        assert page.locator('.project-link').get_attribute('href') == 'https://github.com/cribbinsb/nanocodec'
        # Only the default clip loads on page open: no other clip file was requested.
        fetched = page.evaluate("performance.getEntriesByType('resource').map(e => e.name).filter(n => n.includes('/clips/') && n.endsWith('.mp4'))")
        assert all(n.endswith('/clips/netflix-bar-scene-720p.mp4') for n in fetched), fetched
        missing = page.evaluate("""import('./clips.js').then(({ CLIPS, clipUrl }) => Promise.all(CLIPS.flatMap(c =>
          [360, 720].map(h => fetch(clipUrl(c, h), { method: 'HEAD' }).then(r => r.ok ? null : clipUrl(c, h)))))
          ).then(urls => urls.filter(Boolean))""")
        assert not missing, missing
        # The picker groups its clips (clips.js CLIP_GROUPS).
        assert page.evaluate("[...document.querySelectorAll('#clip optgroup')].map(g => g.label)") == ['People', 'Motion', 'Hard texture', 'Animation']
        for clip_id, resolution, variant in (('svt-waterfall', '640x360', '360p'), ('netflix-tango', '640x360', '360p'), ('svt-smoke-sauna', '1280x720', '720p')):
            for picker in page.locator('.resolution').all():
                picker.select_option(resolution)
            page.locator('#clip').select_option(clip_id)
            page.wait_for_function(f"{clip_src}.endsWith('/clips/{clip_id}-{variant}.mp4')", timeout=10000)
            width = int(resolution.split('x')[0])
            before = page.evaluate('window.workerFrames.length')
            page.wait_for_function(f'window.workerFrames.slice({before}).filter(f => f.width === {width}).length >= 4', timeout=60000)
            # Views at the same frame rate show the same source instant: captures are
            # released together, in order (the badges carry the tick and its time).
            for _ in range(3):
                ticks = page.evaluate("[...document.querySelectorAll('.frame-badge')].map(b => b.dataset.tick)")
                assert ticks[0] == ticks[1], ticks
                page.wait_for_timeout(100)
        # Advanced (collapsed by default) offers what each codec supports: A is H.264, B AV1.
        assert page.evaluate("[...document.querySelectorAll('details.advanced')].every(d => !d.open)")
        for i in range(2):
            page.locator('details.advanced > summary').nth(i).click()
        page.wait_for_function("[...document.querySelectorAll('.advanced-toggles')].every(t => !t.hidden)", timeout=30000)
        offered = page.evaluate("""[...document.querySelectorAll('.advanced')].map(a =>
          [...a.querySelectorAll('[data-option]')].filter(l => !l.hidden).map(l => l.dataset.option))""")
        assert offered == [['blockQp', 'temporalLayers', 'secondReference'], ['filmGrain', 'blockQp', 'temporalLayers']], offered
        # Film grain and one parameter override on B: a bad line shows an inline error and
        # is not applied; a good one restarts B with it over the preset.
        page.evaluate('window.workerConfigs.length = 0')
        before = page.evaluate('window.workerFrames.length')
        page.locator('.film-grain').nth(1).check()
        page.wait_for_function("window.workerConfigs.some(c => c.codec === 'av1' && c.filmGrain)", timeout=10000)
        overrides = page.locator('.param-text').nth(1)
        overrides.fill('nope.nothing=1')
        overrides.dispatch_event('change')
        page.wait_for_function("document.querySelectorAll('.param-errors')[1].textContent.includes('Line 1: unknown parameter')")
        overrides.fill('av1.chroma_delta_q = -12')
        overrides.dispatch_event('change')
        page.wait_for_function("window.workerConfigs.some(c => c.codec === 'av1' && c.filmGrain && c.overrides === 'av1.chroma_delta_q=-12')", timeout=10000)
        assert page.evaluate("document.querySelectorAll('.param-errors')[1].textContent") == ''
        page.wait_for_function(f'window.workerFrames.length >= {before} + 8', timeout=60000)
        assert not any(c['filmGrain'] or c['overrides'] for c in page.evaluate('window.workerConfigs') if c['codec'] != 'av1')
        search = page.locator('.param-search').nth(1)
        search.fill('rate.aq_strength')
        search.dispatch_event('input')
        assert 'range 0 to 3' in page.locator('.param-doc').nth(1).inner_text()
        # The effective settings and the equivalent CLI command.
        page.locator('details.reproduce > summary').nth(1).click()
        page.wait_for_function("document.querySelectorAll('.reproduce .cli')[1].textContent.startsWith('nanocodec encode')")
        command = page.locator('.reproduce .cli').nth(1).inner_text()
        for part in ('nanocodec encode --codec av1', '--bitrate 300', '--param av1.chroma_delta_q=-12', '--film-grain', '--output stream.ivf'):
            assert part in command, command
        assert not errors, errors
        # A pre-encoded reference stream in B (reference.js) against live A at another
        # frame rate. Choosing the stream switches to its clip and fetches only its file;
        # "Match" sets A live to the stream's size, frame rate and bitrate; A then runs
        # at 30 fps against B's 15 fps stream.
        reference_id = 'xiph-johnny-openh264-640x360-15fps-150k'
        page.locator('.source-picker').nth(1).select_option('reference')
        page.wait_for_function(f"[...document.querySelectorAll('.stream-picker')[1].options].some(o => o.value === '{reference_id}')", timeout=10000)
        page.locator('.stream-picker').nth(1).select_option(reference_id)
        page.wait_for_function("document.querySelector('#clip').value === 'xiph-johnny'", timeout=10000)
        page.evaluate('window.workerConfigs.length = 0')
        page.locator('.match').nth(1).click()
        # Each of these settings restarts the comparison (stream file, workers), which
        # can take well over 10 s on a loaded machine: the same bound as the other restarts.
        matched = "window.workerConfigs.some(c => c.codec === 'h264' && c.width === 640 && c.fps === 15 && c.targetKbps === 150 && c.keyInterval === 150)"
        page.wait_for_function(matched, timeout=60000)
        page.locator('.fps').nth(0).select_option('30')
        page.wait_for_function("window.workerConfigs.some(c => c.width === 640 && c.fps === 30)", timeout=60000)
        badge_ticks = "[...document.querySelectorAll('.frame-badge')].map(b => +b.dataset.tick)"
        page.wait_for_function(f"document.querySelectorAll('.frame-badge')[1].textContent.startsWith('PRE-ENCODED') && {badge_ticks}.every(t => t > 30)", timeout=60000)
        assert page.locator('#view-note').is_visible() and 'frame rate' in page.locator('#view-note').inner_text()
        assert 'openh264-harness' in page.locator('.reference-cli').nth(1).inner_text()
        assert 'openh264 v' in page.locator('.reference-version').nth(1).inner_text()
        assert page.locator('[data-metric="encode"]').nth(1).inner_text() == 'n/a'
        assert float(page.locator('[data-metric="psnr"]').nth(1).inner_text()) > 25
        fetched = page.evaluate("performance.getEntriesByType('resource').map(e => e.name).filter(n => n.includes('/streams-ref/') && n.endsWith('.ivf'))")
        # Files are fetched only for streams a comparison started with: the first in the list
        # (picked when the source switched to pre-encoded) and the chosen one.
        first = page.evaluate("import('./reference.js').then(m => m.loadManifest()).then(s => s[0].file)")
        assert f'{reference_id}.ivf' in [n.split('/')[-1] for n in fetched] and {n.split('/')[-1] for n in fetched} <= {first, f'{reference_id}.ivf'}, fetched
        # Pause lands both views on the next instant both code (a 15 fps instant: an even
        # tick); Step moves both to the next one.
        page.locator('#pause').click()
        page.wait_for_function("document.querySelector('#pause').textContent === 'Resume' && window.framesInFlight() === 0", timeout=30000)
        page.wait_for_timeout(300)
        paused_ticks = page.evaluate(badge_ticks)
        assert paused_ticks[0] == paused_ticks[1] and paused_ticks[0] % 2 == 0, paused_ticks
        page.locator('#step').click()
        page.wait_for_function(f"{badge_ticks}.every(t => t === {paused_ticks[0] + 2})", timeout=30000)
        page.locator('#pause').click()
        page.locator('.source-picker').nth(1).select_option('live')
        # B restarts live (its frames resume) before the webcam run is configured.
        page.wait_for_function("document.querySelectorAll('.frame-badge')[1].textContent.startsWith('WASM DECODED')", timeout=60000)
        # A back at the default rates for the webcam run.
        for selector, kbps in (('.peak-rate', '600'), ('.target-rate', '300')):
            page.locator(selector).nth(0).fill(kbps)
            page.locator(selector).nth(0).dispatch_event('change')
        assert not errors, errors
        page.locator('#start').click()  # stop the clip before configuring the webcam run
        page.evaluate('window.workerFrames.length = 0; window.workerConfigs.length = 0')
        page.locator('.codec-picker').nth(0).select_option('vp9')
        page.locator('.codec-picker').nth(1).select_option('av2')
        for picker in page.locator('.resolution').all():
            picker.select_option('384x216')
        for picker in page.locator('.fps').all():
            picker.select_option('10')
        for picker in page.locator('.gop').all():
            picker.select_option('30')
        page.locator('[data-source="camera"]').click()
        page.locator('#start').click()
        page.wait_for_function('window.workerFrames.length >= 16', timeout=60000)
        assert not errors, errors
        assert page.locator('.rate-status').nth(0).inner_text().startswith('Target 300 kbps')
        page.locator('#pause').click()
        # Pausing stops capture at the next instant both views code; the frames already
        # sent still complete, however long that takes under load. Once they have,
        # nothing more arrives.
        page.wait_for_function("document.querySelector('#pause').textContent === 'Resume' && window.framesInFlight() === 0", timeout=60000)
        # The counts catch up with the last frame within panel.js METRICS_INTERVAL_MS (250 ms).
        page.wait_for_timeout(300)
        frozen = page.locator('.frame-count').all_inner_texts()
        page.wait_for_timeout(600)
        assert page.locator('.frame-count').all_inner_texts() == frozen
        assert page.evaluate('window.framesInFlight()') == 0
        # A key frame requested while paused is the next frame each view codes.
        before = page.evaluate('window.workerFrames.length')
        page.locator('#keyframe').click()
        page.locator('#pause').click()
        first_after_key = f'''new Map(window.workerFrames.slice({before}).reverse().map(f => [f.view, f.key]))'''
        page.wait_for_function(f'{first_after_key}.size === 2', timeout=60000)
        assert page.evaluate(f'[...{first_after_key}.values()]') == [True, True]
        page.locator('.target-rate').nth(0).fill('300')
        page.locator('.target-rate').nth(0).dispatch_event('change')
        page.wait_for_function("document.querySelector('.rate-status').textContent.startsWith('Target 300 kbps')")
        page.locator('#pause').click()
        page.wait_for_timeout(500)
        streams = []
        for i, fourcc in enumerate((b'VP90', b'AV02')):
            with page.expect_download() as event:
                page.locator('.download').nth(i).click()
            download = event.value
            data = Path(download.path()).read_bytes()
            assert data[:4] == b'DKIF' and data[8:12] == fourcc
            assert struct.unpack_from('<HH', data, 12) == (384, 216)
            offset, count, last = 32, 0, -1
            while offset < len(data):
                size, timestamp = struct.unpack_from('<IQ', data, offset)
                offset += 12
                assert size and offset + size <= len(data) and timestamp > last
                offset += size
                count += 1
                last = timestamp
            assert offset == len(data) and count >= 8
            streams.append({'codec':fourcc.decode(), 'frames':count, 'bytes':len(data)})
            download.delete()
        # VVC in both views (A Default, B Fast) on the built-in scene, drawn with WebGL2:
        # stats, hold-original, magnifier and the Annex-B downloads.
        page.locator('#start').click()  # stop the camera run
        page.evaluate('window.workerFrames.length = 0')
        for picker in page.locator('.codec-picker').all():
            picker.select_option('vvc')
        page.locator('.preset').nth(1).select_option('fast')
        page.locator('[data-source="scene"]').click()
        page.locator('#start').click()
        page.wait_for_function('window.workerFrames.length >= 12', timeout=60000)
        assert page.evaluate("window.workerFrames.every(f => f.format === 'i420' && f.width === 384 && f.height === 216)")
        assert [(c['codec'], c['preset']) for c in page.evaluate('window.workerConfigs')[-2:]] == [('vvc', 'default'), ('vvc', 'fast')]
        assert all(float(t) > 20 for t in page.locator('[data-metric="psnr"]').all_inner_texts())
        assert all(t.startswith('Target 300 kbps · QP') for t in page.locator('.rate-status').all_inner_texts())
        page.locator('#original').focus()
        page.keyboard.down(' ')
        page.wait_for_function("[...document.querySelectorAll('.frame-badge')].every(b => b.textContent.startsWith('ORIGINAL SOURCE'))", timeout=10000)
        page.keyboard.up(' ')
        page.wait_for_function("[...document.querySelectorAll('.frame-badge')].every(b => b.textContent.startsWith('WASM DECODED'))", timeout=10000)
        page.locator('.loupe-toggle').click()
        page.locator('.viewport canvas').nth(0).click(position={'x': 100, 'y': 60})
        page.wait_for_timeout(300)
        assert page.evaluate("""[...document.querySelectorAll('.loupe-panes canvas')].every(c =>
          c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v))""")
        page.locator('.loupe-toggle').click()
        page.locator('#pause').click()
        page.wait_for_timeout(500)
        for i in range(2):
            with page.expect_download() as event:
                page.locator('.download').nth(i).click()
            download = event.value
            assert download.suggested_filename.endswith('-vvc.vvc')
            data = Path(download.path()).read_bytes()
            # H.266 NAL unit types (§7.4.2.2): the key access unit opens with SPS (15)
            # and PPS (16); IDR_N_LP (8) and TRAIL (0) pictures follow.
            types = [data[i + 4] >> 3 for i in range(len(data) - 4) if data[i:i + 3] == b'\0\0\1']
            assert data[:4] == b'\0\0\0\1' and types[:2] == [15, 16] and 8 in types
            pictures = sum(t < 12 for t in types)
            assert pictures >= 8
            streams.append({'codec':'VVC', 'frames':pictures, 'bytes':len(data)})
            download.delete()
        page.locator('#pause').click()
        for picker in page.locator('.resolution').all():
            picker.select_option('640x360')
        page.wait_for_function('window.workerFrames.some(x=>x.width===640 && x.height===360)', timeout=60000)
        assert page.evaluate('window.workerConfigs.length>=4 && window.workerConfigs.every(c=>c.keyInterval===300)')
        # The queue stays bounded (pipeline.js MAX_FRAMES_IN_FLIGHT).
        max_in_flight = page.evaluate('window.maxInFlight')
        assert 1 <= max_in_flight <= 4, max_in_flight
        summary = {'status':'pass','max_frames_in_flight':max_in_flight,'browser':browser.version,'source':'Chromium synthetic webcam','autostart_clip':True,'startup_lookahead':True,
                   'streams':streams,'decoded_frames':page.evaluate('window.workerFrames.length'),
                   # Frames the encoder held as repeats (frame hold): shown, with nothing coded.
                   'repeated_frames':page.evaluate('window.repeatFrames || 0'),
                   'pause_resume':True,'forced_keys':True,'live_cbr_change':True,'gop_seconds':30,
                   'resolution_recreation':True,'pre_encoded_view':True,'pause_on_common_instant':True,'webgl2_display':True,'vvc_scene_hold_magnifier':True,'av1_film_grain':True,'advanced_overrides':True,'clip_variants':True}
        # A GL context that stays lost falls back to the CPU path on a fresh canvas (VVC in
        # both views here); the magnifier follows it.
        page.evaluate("document.querySelector('.viewport canvas').getContext('webgl2').getExtension('WEBGL_lose_context').loseContext()")
        page.wait_for_function("document.querySelector('.viewport').dataset.display === 'cpu'")
        before = page.evaluate('window.workerFrames.length')
        page.wait_for_function(f"window.workerFrames.slice({before}).filter(f => f.format === 'rgba').length >= 2", timeout=60000)
        page.locator('.loupe-toggle').click()
        page.locator('.viewport').nth(0).click(position={'x': 100, 'y': 60})
        page.wait_for_timeout(300)
        assert page.evaluate("""[...document.querySelectorAll('.loupe-panes canvas')].every(c =>
          c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some((v, i) => i % 4 === 3 && v))""")
        assert not errors, errors
        context.close()
        browser.close()
        # Automatic choice: software GL (SwiftShader, the default here) and no WebGL both give the CPU path.
        for flags in ([], ['--disable-webgl']):
            browser = p.chromium.launch(headless=True, args=flags)
            page = browser.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/')
            page.wait_for_function("[...document.querySelectorAll('.frame-badge')].every(b => b.textContent.includes('#3'))", timeout=60000)
            assert page.evaluate("[...document.querySelectorAll('.viewport')].map(v => v.dataset.display)") == ['cpu', 'cpu']
            assert not errors, errors
            browser.close()
        print(json.dumps(summary | {'context_loss_fallback':True,'software_gl_fallback':True,'no_webgl_fallback':True}))
except Exception:
    print('page errors:', errors, file=sys.stderr)
    print('last page notices, session lines and lifecycle trace:', *notices[-40:], sep='\n  ', file=sys.stderr)
    raise
finally:
    server.shutdown()
    server.server_close()
