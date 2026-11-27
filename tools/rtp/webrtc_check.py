#!/usr/bin/env python3
"""WebRTC end to end: our RTP packets decoded by Chrome's own WebRTC stack.

For each case, encode a short synthetic clip with `nanocodec encode --rtp`,
send those packets (payload bytes unchanged; payload type, SSRC and the
Dependency Descriptor's extension id rewritten to the negotiated ones) from
a Pion peer (tools/rtp/pion `webrtc`) to headless Google Chrome over a real
WebRTC session (ICE, DTLS-SRTP) on the loopback, and require every picture
Chrome decodes to equal our stream's pictures (SHA-256 of the I420 planes
from MediaStreamTrackProcessor / VideoFrame.copyTo), all of them, in order.

VP9 spatial SVC cases (`vp9-L2T1`, ...) send the whole stream, or, with a
`-S<n>` suffix, the packets an SFU forwards to a receiver of spatial layer n
(`nanocodec depacketize --layers n:0 --forward`, by the VP9 payload
descriptor alone); Chrome must show that layer's pictures exactly as vpxdec
decodes `nanocodec extract` of it (one picture per unit: the top frame).

usage: webrtc_check.py --nanocodec BIN --pion BIN [--cases av1,h264,vp9,...]
                       [--chrome /usr/bin/google-chrome] [--work DIR] [--keep]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interop import FPS, HEIGHT, QP, WIDTH, make_source  # noqa: E402

FRAMES = 60
KEY_INTERVAL = 30
FFMPEG = "/usr/bin/ffmpeg"
VPXDEC = Path.home() / "nanocodec-data/anchors/x86_64/bin/vpxdec"
PORT = 8093
CASES = {  # name: (codec, extra encode options)
    "av1": ("av1", []),
    "h264": ("h264", []),
    "vp9": ("vp9", []),
    "av1-L1T3": ("av1", ["--temporal-layers", "3", "--param", "layers.adaptive=0"]),
    "av1-L2T2": ("av1", ["--scalability", "L2T2"]),
    "vp9-L1T2": ("vp9", ["--temporal-layers", "2", "--param", "layers.adaptive=0"]),
}
#: VP9 spatial SVC: scalability mode -> spatial layer count. Each gives a
#: case for the whole stream and one per forwarded lower layer (`-S<n>`).
VP9_SVC_MODES = {"L2T1": 2, "L3T1": 3, "L2T1_KEY": 2, "L3T1_KEY": 3}
for _mode, _layers in VP9_SVC_MODES.items():
    CASES[f"vp9-{_mode}"] = ("vp9", ["--scalability", _mode])
    for _s in range(_layers - 1):
        CASES[f"vp9-{_mode}-S{_s}"] = ("vp9", ["--scalability", _mode])
CHROME_FLAGS = [
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--autoplay-policy=no-user-gesture-required", "--mute-audio",
    # Chrome advertises the Dependency Descriptor only behind this trial.
    "--force-fieldtrials=WebRTC-DependencyDescriptorAdvertised/Enabled/",
]


def run(cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{result.stderr[-2000:]}")
    return result


def reference_hashes(stream, width, height):
    """SHA-256 of each decoded picture's tight I420 planes (ffmpeg)."""
    data = subprocess.run([FFMPEG, "-v", "error", "-i", str(stream), "-f", "rawvideo",
                           "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    size = width * height * 3 // 2
    return [hashlib.sha256(data[i:i + size]).hexdigest() for i in range(0, len(data), size)]


def vpxdec_hashes(stream, width, height):
    """SHA-256 of each picture vpxdec shows (one per unit: libvpx, as
    Chrome's VP9 decoder, outputs a superframe's last frame)."""
    data = subprocess.run([VPXDEC, "--codec=vp9", "--i420", "--rawvideo", "-o", "-", str(stream)],
                          capture_output=True, check=True).stdout
    size = width * height * 3 // 2
    return [hashlib.sha256(data[i:i + size]).hexdigest() for i in range(0, len(data), size)]


def vp9_svc_reference(name, nc, stream, capture, work):
    """The capture to send and Chrome's expected pictures for a VP9 SVC case:
    the whole stream, or the packets forwarded to spatial layer S (`-S<n>`)."""
    mode, _, forward = name.removeprefix("vp9-").partition("-S")
    layers = VP9_SVC_MODES[mode]
    spatial = int(forward) if forward else layers - 1
    scale = 1 << (layers - 1 - spatial)
    width, height = WIDTH // scale, HEIGHT // scale
    if not forward:
        return capture, vpxdec_hashes(stream, width, height)
    sent = work / f"{name}-forwarded.rtp"
    key_only = ["--inter-layer", "key"] if mode.endswith("_KEY") else []
    run([nc, "depacketize", "--codec", "vp9", "--input", capture, "--layers", f"{spatial}:0",
         *key_only, "--forward", sent, "--output", work / f"{name}-forwarded.ivf"])
    expected = work / f"{name}-extract.ivf"
    run([nc, "extract", "--codec", "vp9", "--input", stream,
         "--operating-point", layers - 1 - spatial, "--output", expected])
    return sent, vpxdec_hashes(expected, width, height)


def h264_fmtp(stream):
    """The SDP fmtp of an H.264 stream: profile_idc, constraint flags and
    level_idc from its first SPS (H.264 §7.3.2.1.1)."""
    data = Path(stream).read_bytes()
    at = data.index(b"\x00\x00\x01\x67") + 4
    profile_level = data[at:at + 3].hex()
    return f"level-asymmetry-allowed=1;packetization-mode=1;profile-level-id={profile_level}"


def check_case(name, nc, pion, chrome, work):
    codec, extra = CASES[name]
    stream = work / f"{name}.{'264' if codec == 'h264' else 'ivf'}"
    capture = work / f"{name}.rtp"
    run([nc, "encode", "--codec", codec, "--input", work / "source.yuv", "--width", WIDTH,
         "--height", HEIGHT, "--fps", FPS, "--frames", FRAMES, "--qp", QP,
         "--intra-period", KEY_INTERVAL, *extra, "--output", stream, "--rtp", capture])
    if "--scalability" in extra and codec == "vp9":
        capture, reference = vp9_svc_reference(name, nc, stream, capture, work)
    else:
        reference = reference_hashes(stream, WIDTH, HEIGHT)
    result_path = work / f"{name}.json"
    fmtp = h264_fmtp(stream) if codec == "h264" else ""
    sender = subprocess.Popen([str(pion), "webrtc", "--codec", codec, "--in", str(capture),
                               "--frames", str(len(reference)), "--out", str(result_path),
                               "--port", str(PORT), "--fmtp", fmtp],
                              stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    profile = work / f"chrome-profile-{name}"
    browser = subprocess.Popen([chrome, *CHROME_FLAGS, f"--user-data-dir={profile}",
                                f"http://127.0.0.1:{PORT}/"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        sender_log = sender.communicate(timeout=90)[1]
    finally:
        browser.kill()
        browser.wait()
    if sender.returncode != 0 or not result_path.exists():
        return False, f"sender failed: {sender_log.strip().splitlines()[-1:]}"
    report = json.loads(result_path.read_text())
    if "error" in report:
        return False, f"page: {report['error']}"
    got = report["hashes"]
    inbound = next((s for s in report["stats"] if s.get("type") == "inbound-rtp"), {})
    codec_stats = next((s for s in report["stats"] if s.get("type") == "codec"), {})
    detail = (f"{len(got)}/{len(reference)} pictures identical={got == reference}; "
              f"decoder={inbound.get('decoderImplementation')} "
              f"framesDecoded={inbound.get('framesDecoded')} "
              f"keyFramesDecoded={inbound.get('keyFramesDecoded')} "
              f"packetsLost={inbound.get('packetsLost')} codec={codec_stats.get('mimeType')} "
              f"formats={report.get('formats')}")
    dd = [line for line in report.get("answerExtmaps", []) if "dependency-descriptor" in line]
    if dd:
        detail += f"; {dd[0].strip()}"
    return got == reference, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nanocodec", required=True)
    parser.add_argument("--pion", required=True)
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--work", default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    work = Path(args.work or Path.home() / "nanocodec-data/scratch/rtp/webrtc").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    make_source(work / "source.yuv", FRAMES)
    nc, pion = Path(args.nanocodec).resolve(), Path(args.pion).resolve()
    failures = 0
    for name in args.cases.split(","):
        try:
            ok, detail = check_case(name, nc, pion, args.chrome, work)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            ok, detail = False, str(error).strip().splitlines()[-1][:300]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  webrtc {name}  {detail}", flush=True)
    if not args.keep:
        shutil.rmtree(work)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
