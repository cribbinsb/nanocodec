// The WebRTC end-to-end check (docs/handoff/rtp.md): serve a page that
// opens a receive-only RTCPeerConnection, answer it with Pion (pion/webrtc,
// MIT licence), send nanocodec's own RTP packets (read from an RFC 4571
// file, paced by their timestamps) on the negotiated track, and collect what
// the page reports: a hash of every picture Chrome's WebRTC stack decoded
// (MediaStreamTrackProcessor, VideoFrame.copyTo as I420) and the receiver's
// getStats. Only the payload type, SSRC and header extension id are
// rewritten to the negotiated ones; payloads go out byte for byte.
//
//	rtpinterop webrtc --codec av1|h264|vp9 --in OURS.rtp --frames N
//	                  --out RESULT.json [--port 8090] [--fmtp F]
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

const dependencyDescriptorURI = "https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension"

// Our packets carry the descriptor as extension id 1 (nanocodec's default).
const ourDescriptorID = 1

func runWebRTC(args []string) error {
	flags := flag.NewFlagSet("webrtc", flag.ExitOnError)
	codec := flags.String("codec", "", "av1, h264 or vp9")
	in := flags.String("in", "", "RFC 4571 file of nanocodec's RTP packets")
	frames := flags.Int("frames", 0, "pictures the stream holds")
	out := flags.String("out", "", "result JSON")
	port := flags.Int("port", 8090, "HTTP port")
	fmtp := flags.String("fmtp", "", "SDP fmtp line of the codec (H.264 profile)")
	timeout := flags.Duration("timeout", 60*time.Second, "give up after")
	_ = flags.Parse(args)

	packets, err := readFramed(*in)
	if err != nil {
		return err
	}
	mime := map[string]string{"av1": webrtc.MimeTypeAV1, "h264": webrtc.MimeTypeH264, "vp9": webrtc.MimeTypeVP9}[*codec]
	if mime == "" {
		return fmt.Errorf("unknown codec %q", *codec)
	}
	capability := webrtc.RTPCodecCapability{MimeType: mime, ClockRate: clockRate, SDPFmtpLine: *fmtp}
	media := &webrtc.MediaEngine{}
	if err := media.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: capability,
		PayloadType:        payloadType,
	}, webrtc.RTPCodecTypeVideo); err != nil {
		return err
	}
	if err := media.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: dependencyDescriptorURI},
		webrtc.RTPCodecTypeVideo); err != nil {
		return err
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(media))

	results := make(chan []byte, 1)
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, receiverPage(*frames))
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		answer, err := answerOffer(api, r.Body, capability, packets)
		if err != nil {
			fmt.Fprintln(os.Stderr, "webrtc:", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_, _ = w.Write(answer)
	})
	mux.HandleFunc("/result", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		results <- body
	})
	server := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", *port), Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = server.ListenAndServe() }()
	defer server.Close()
	fmt.Fprintf(os.Stderr, "webrtc: serving http://127.0.0.1:%d/ with %d packets\n", *port, len(packets))
	select {
	case body := <-results:
		return os.WriteFile(*out, body, 0o644)
	case <-time.After(*timeout):
		return errors.New("no result from the page in time")
	}
}

func readFramed(path string) ([][]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var packets [][]byte
	for at := 0; at+2 <= len(data); {
		n := int(binary.BigEndian.Uint16(data[at:]))
		if at+2+n > len(data) {
			return nil, errors.New("truncated RFC 4571 file")
		}
		packets = append(packets, data[at+2:at+2+n])
		at += 2 + n
	}
	return packets, nil
}

// answerOffer answers the page's offer and starts sending once connected.
func answerOffer(api *webrtc.API, body io.Reader, capability webrtc.RTPCodecCapability,
	packets [][]byte) ([]byte, error) {
	var offer webrtc.SessionDescription
	if err := json.NewDecoder(body).Decode(&offer); err != nil {
		return nil, err
	}
	pc, err := api.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	track, err := webrtc.NewTrackLocalStaticRTP(capability, "video", "nanocodec")
	if err != nil {
		return nil, err
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		return nil, err
	}
	go func() { // drain RTCP
		buf := make([]byte, 1500)
		for {
			if _, _, err := sender.Read(buf); err != nil {
				return
			}
		}
	}()
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		fmt.Fprintln(os.Stderr, "webrtc: connection", state)
		if state == webrtc.PeerConnectionStateConnected {
			go send(track, sender, packets)
		}
	})
	if err := pc.SetRemoteDescription(offer); err != nil {
		return nil, err
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return nil, err
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		return nil, err
	}
	<-gathered
	return json.Marshal(pc.LocalDescription())
}

// send writes our packets at the pace of their timestamps, with the
// descriptor moved to the negotiated extension id (dropped if none).
func send(track *webrtc.TrackLocalStaticRTP, sender *webrtc.RTPSender, packets [][]byte) {
	negotiated := 0
	for _, ext := range sender.GetParameters().HeaderExtensions {
		if ext.URI == dependencyDescriptorURI {
			negotiated = ext.ID
		}
	}
	fmt.Fprintln(os.Stderr, "webrtc: dependency descriptor id", negotiated)
	time.Sleep(500 * time.Millisecond) // let the page's track and processor start
	start := time.Now()
	var first uint32
	for index, raw := range packets {
		var packet rtp.Packet
		if err := packet.Unmarshal(bytes.Clone(raw)); err != nil {
			fmt.Fprintln(os.Stderr, "webrtc:", err)
			return
		}
		if index == 0 {
			first = packet.Timestamp
		}
		due := time.Duration(packet.Timestamp-first) * time.Second / clockRate
		time.Sleep(time.Until(start.Add(due)))
		descriptor := packet.GetExtension(ourDescriptorID)
		descriptor = bytes.Clone(descriptor)
		_ = packet.DelExtension(ourDescriptorID)
		if negotiated != 0 && descriptor != nil {
			if err := packet.SetExtension(uint8(negotiated), descriptor); err != nil {
				fmt.Fprintln(os.Stderr, "webrtc:", err)
				return
			}
		}
		if err := track.WriteRTP(&packet); err != nil {
			fmt.Fprintln(os.Stderr, "webrtc:", err)
			return
		}
	}
	fmt.Fprintln(os.Stderr, "webrtc: sent", len(packets), "packets")
}

// receiverPage is the browser side: receive, hash every decoded picture,
// report.
func receiverPage(frames int) string {
	return `<!doctype html><meta charset="utf-8"><title>nanocodec WebRTC receive</title>
<pre id="log"></pre><script>
const expected = ` + fmt.Sprint(frames) + `;
const log = (m) => { document.getElementById('log').textContent += m + '\n'; };
async function hash(frame) {
  // The decoder's own planes, tightly packed (software decoders give I420).
  if (frame.format !== 'I420') throw new Error('format ' + frame.format);
  const buffer = new Uint8Array(frame.allocationSize());
  await frame.copyTo(buffer);
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
}
async function main() {
  const pc = new RTCPeerConnection();
  pc.addTransceiver('video', {direction: 'recvonly'});
  const hashes = [];
  let formats = new Set();
  const done = new Promise((resolve) => {
    pc.ontrack = async (event) => {
      const reader = new MediaStreamTrackProcessor({track: event.track, maxBufferSize: 64}).readable.getReader();
      const deadline = Date.now() + 30000;
      while (hashes.length < expected && Date.now() < deadline) {
        const result = await Promise.race([reader.read(), new Promise(r => setTimeout(() => r({timeout: true}), 3000))]);
        if (result.timeout) break;
        if (result.done) break;
        const frame = result.value;
        formats.add(frame.format + ' ' + frame.visibleRect.width + 'x' + frame.visibleRect.height);
        try { hashes.push(await hash(frame)); } catch (e) { hashes.push('error: ' + e); }
        frame.close();
      }
      resolve();
    };
  });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise(r => { if (pc.iceGatheringState === 'complete') r(); else pc.onicegatheringstatechange = () => pc.iceGatheringState === 'complete' && r(); });
  const answer = await (await fetch('/offer', {method: 'POST', body: JSON.stringify(pc.localDescription)})).json();
  await pc.setRemoteDescription(answer);
  await done;
  const stats = [];
  (await pc.getStats()).forEach(s => { if (s.type === 'inbound-rtp' || s.type === 'codec') stats.push(s); });
  const report = {hashes, formats: [...formats], stats, userAgent: navigator.userAgent,
                  answerExtmaps: answer.sdp.split('\n').filter(l => l.startsWith('a=extmap') || l.startsWith('a=rtpmap') || l.startsWith('a=fmtp'))};
  log(JSON.stringify(report, null, 1));
  await fetch('/result', {method: 'POST', body: JSON.stringify(report)});
}
main().catch(e => fetch('/result', {method: 'POST', body: JSON.stringify({error: String(e)})}));
</script>`
}
