#!/usr/bin/env python3
"""RTP interop checks of nanocodec-rtp against independent implementations.

For each codec: encode a short synthetic clip with `nanocodec encode --rtp`,
then carry it over RTP every way below and require the decoded pictures to
equal the stream's own, bit for bit (ffmpeg's decoders, `-f framemd5`):

  ours -> ours          nanocodec depacketize
  ours -> GStreamer     rtp*depay from the RFC 4571 file and from the pcap
  ours -> Pion          pion/rtp depacketizers (tools/rtp/pion)
  GStreamer -> ours     rtp*pay output into nanocodec depacketize
  Pion -> ours          pion/rtp payloaders into nanocodec depacketize
  reorder               packets shuffled within 8 -> identical pictures
  loss                  one packet of a P frame dropped -> the frames before
                        it shown exact, decoding resumes exactly at the next
                        key frame (and a keyframe request is the signal)
  random loss           3 % of packets dropped -> every shown picture is one
                        of the stream's, in order (no corrupt picture)
  top layer lost        (H.264, AV1, VP9 with two temporal layers) a whole
                        top-temporal-layer frame lost -> every other frame
                        shown exact, no wait for a key frame
  svc (AV1)             L2T2 / L3T1 / L2T1 K-SVC: every lower operating point
                        forwarded by the Dependency Descriptor alone
                        (`depacketize --layers S:T`) decodes, in dav1d,
                        exactly as `nanocodec extract` of that operating
                        point does

GStreamer has no AV1 or VVC payloader in 1.28's good/bad plugin sets, so
those two codecs are checked against Pion only. Light: under a minute.

usage: interop.py --nanocodec BIN --pion BIN [--codecs h264,hevc,vvc,av1,vp9]
                  [--work DIR] [--keep]
"""

import argparse
import hashlib
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

WIDTH, HEIGHT, FRAMES, FPS = 352, 288, 40, 30
KEY_INTERVAL = 15
QP = 28
GST = "/usr/bin/gst-launch-1.0"
FFMPEG = "/usr/bin/ffmpeg"
REORDER_SPAN = 8
LOSS_RATE = 0.03
SEED = 7
CODECS = ("h264", "hevc", "vvc", "av1", "vp9")
EXTENSION = {"h264": "264", "hevc": "265", "vvc": "266", "av1": "ivf", "vp9": "ivf"}
FFMPEG_FORMAT = {"h264": "h264", "hevc": "hevc", "vvc": "vvc"}
GST_NAMES = {  # codec: (encoding-name, payloader, depayloader, parse, caps)
    "h264": ("H264", "rtph264pay", "rtph264depay", "h264parse",
             "video/x-h264,stream-format=byte-stream,alignment=au"),
    "hevc": ("H265", "rtph265pay", "rtph265depay", "h265parse",
             "video/x-h265,stream-format=byte-stream,alignment=au"),
    "vp9": ("VP9", "rtpvp9pay", "rtpvp9depay", None, None),
}
SVC_CASES = (  # scalability mode, operating points to forward as (op, S, T)
    ("L2T2", ((1, 1, 1), (2, 0, 1), (3, 0, 0))),
    ("L3T1", ((1, 1, 0), (2, 0, 0))),
    ("L2T1_KEY", ((1, 0, 0),)),
)


def run(cmd, **kw):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{result.stderr[-2000:]}")
    return result


def make_source(path, frames=FRAMES):
    """A textured picture panning two pixels a frame: content that codes to
    several packets a frame, deterministic."""
    rng = random.Random(SEED)
    big_w, big_h = WIDTH + 2 * frames, HEIGHT + frames
    texture = bytes((((x * 7) ^ (y * 5)) + rng.randrange(24)) & 0xFF
                    for y in range(big_h) for x in range(big_w))
    with open(path, "wb") as out:
        for t in range(frames):
            for y in range(HEIGHT):
                row = (y + t) * big_w + 2 * t
                out.write(texture[row:row + WIDTH])
            out.write(bytes([128]) * (WIDTH * HEIGHT // 2))


def frame_hashes(codec, stream):
    cmd = [FFMPEG, "-v", "error"]
    if codec in FFMPEG_FORMAT:
        cmd += ["-f", FFMPEG_FORMAT[codec]]
    cmd += ["-i", stream, "-f", "framemd5", "-"]
    lines = run(cmd).stdout.splitlines()
    return [line.split(",")[-1].strip() for line in lines if line and not line.startswith("#")]


def read_framed(path):
    data = Path(path).read_bytes()
    packets, at = [], 0
    while at < len(data):
        (length,) = struct.unpack(">H", data[at:at + 2])
        packets.append(data[at + 2:at + 2 + length])
        at += 2 + length
    return packets


def write_framed(path, packets):
    with open(path, "wb") as out:
        for packet in packets:
            out.write(struct.pack(">H", len(packet)) + packet)


def units_by_timestamp(packets):
    """Packet indices per unit (RTP timestamp), in order."""
    units = {}
    for index, packet in enumerate(packets):
        units.setdefault(packet[4:8], []).append(index)
    return list(units.values())


def is_subsequence(part, whole):
    it = iter(whole)
    return all(any(h == w for w in it) for h in part)


class Checks:
    def __init__(self):
        self.rows = []

    def record(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    def attempt(self, name, func):
        try:
            ok, detail = func()
        except (RuntimeError, OSError, ValueError) as error:
            ok, detail = False, str(error).strip().splitlines()[-1][:300]
        self.record(name, ok, detail)


def gst_depay(codec, capture, out):
    name, _, depay, parse, caps = GST_NAMES[codec]
    rtp_caps = f"application/x-rtp,media=video,clock-rate=90000,encoding-name={name},payload=96"
    if str(capture).endswith(".pcap"):
        source = ["filesrc", f"location={capture}", "!", "pcapparse", "!", rtp_caps]
    else:
        source = ["filesrc", f"location={capture}", "!",
                  f"application/x-rtp-stream,media=video,clock-rate=90000,encoding-name={name}",
                  "!", "rtpstreamdepay", "!", rtp_caps]
    if codec == "vp9":
        sink = [depay, "!", "avmux_ivf", "!", "filesink", f"location={out}"]
    else:
        sink = [depay, "!", parse, "!", caps, "!", "filesink", f"location={out}"]
    run([GST, "-q", *source, "!", *sink])


def gst_pay(codec, stream, out):
    _, pay, _, parse, caps = GST_NAMES[codec]
    if codec == "vp9":
        head = ["filesrc", f"location={stream}", "!", "ivfparse"]
        payloader = [pay, "mtu=1200"]
    else:
        head = ["filesrc", f"location={stream}", "!", parse, "!", caps]
        payloader = [pay, "mtu=1200", "aggregate-mode=zero-latency", "config-interval=-1"]
    run([GST, "-q", *head, "!", *payloader, "!", "rtpstreampay", "!",
         "filesink", f"location={out}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nanocodec", required=True)
    parser.add_argument("--pion", required=True)
    parser.add_argument("--codecs", default=",".join(CODECS))
    parser.add_argument("--work", default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    nc, pion = Path(args.nanocodec).resolve(), Path(args.pion).resolve()
    work = Path(args.work or Path.home() / "nanocodec-data/scratch/rtp/interop").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    source = work / "source.yuv"
    make_source(source)
    checks = Checks()
    encode_common = ["--input", source, "--width", WIDTH, "--height", HEIGHT, "--fps", FPS,
                     "--frames", FRAMES, "--qp", QP, "--intra-period", KEY_INTERVAL]

    def depacketize(codec, capture, out, *extra):
        run([nc, "depacketize", "--codec", codec, "--input", capture, "--output", out, *extra])
        return out

    for codec in args.codecs.split(","):
        ext = EXTENSION[codec]
        stream = work / f"{codec}.{ext}"
        ours = work / f"{codec}.rtp"
        pcap = work / f"{codec}.pcap"
        run([nc, "encode", "--codec", codec, *encode_common, "--output", stream, "--rtp", ours])
        run([nc, "encode", "--codec", codec, *encode_common, "--output", work / f"{codec}-2.{ext}",
             "--rtp", pcap])
        reference = frame_hashes(codec, stream)
        packets = read_framed(ours)
        print(f"-- {codec}: {len(reference)} frames, {len(packets)} packets, "
              f"largest {max(map(len, packets))} bytes", flush=True)

        def same(path, label=""):
            got = frame_hashes(codec, path)
            return got == reference, f"{len(got)}/{len(reference)} frames identical {label}"

        checks.attempt(f"{codec} ours->ours", lambda: same(
            depacketize(codec, ours, work / f"{codec}-self.{ext}")))
        checks.attempt(f"{codec} ours(pcap)->ours", lambda: same(
            depacketize(codec, pcap, work / f"{codec}-selfpcap.{ext}")))
        checks.attempt(f"{codec} packet size", lambda: (
            all(len(p) <= 1200 for p in packets), "every packet <= 1200 bytes"))
        if codec in GST_NAMES:
            def ours_to_gst(capture, tag):
                out = work / f"{codec}-gst{tag}.{ext}"
                gst_depay(codec, capture, out)
                return same(out)
            checks.attempt(f"{codec} ours->GStreamer", lambda: ours_to_gst(ours, ""))
            checks.attempt(f"{codec} ours(pcap)->GStreamer", lambda: ours_to_gst(pcap, "pcap"))

            def gst_to_ours():
                capture = work / f"{codec}-gstpay.rtp"
                gst_pay(codec, stream, capture)
                return same(depacketize(codec, capture, work / f"{codec}-fromgst.{ext}"))
            checks.attempt(f"{codec} GStreamer->ours", gst_to_ours)

        def ours_to_pion():
            out = work / f"{codec}-pion.{ext}"
            run([pion, "depay", "--codec", codec, "--in", ours, "--out", out])
            return same(out)
        checks.attempt(f"{codec} ours->Pion", ours_to_pion)

        def pion_to_ours(flexible=False):
            capture = work / f"{codec}-pionpay{int(flexible)}.rtp"
            extra = ["--vp9-flexible"] if flexible else []
            run([pion, "pay", "--codec", codec, "--in", stream, "--out", capture, *extra])
            return same(depacketize(codec, capture, work / f"{codec}-frompion.{ext}"))
        checks.attempt(f"{codec} Pion->ours", pion_to_ours)
        if codec == "vp9":
            checks.attempt("vp9 Pion(flexible)->ours", lambda: pion_to_ours(True))

        def reorder():
            rng = random.Random(SEED)
            shuffled = []
            for start in range(0, len(packets), REORDER_SPAN):
                chunk = packets[start:start + REORDER_SPAN]
                rng.shuffle(chunk)
                shuffled += chunk
            capture = work / f"{codec}-reorder.rtp"
            write_framed(capture, shuffled)
            return same(depacketize(codec, capture, work / f"{codec}-reorder.{ext}"), "after reorder")
        checks.attempt(f"{codec} reorder", reorder)

        def loss():
            units = units_by_timestamp(packets)
            lost_unit = KEY_INTERVAL // 3
            victim = units[lost_unit][len(units[lost_unit]) // 2]
            capture = work / f"{codec}-loss.rtp"
            write_framed(capture, [p for i, p in enumerate(packets) if i != victim])
            got = frame_hashes(codec, depacketize(codec, capture, work / f"{codec}-loss.{ext}"))
            expected = reference[:lost_unit] + reference[KEY_INTERVAL:]
            return got == expected, (f"packet {victim} of frame {lost_unit} lost: {len(got)} frames "
                                     f"shown, expected frames 0-{lost_unit - 1} and "
                                     f"{KEY_INTERVAL}-{FRAMES - 1}")

        def random_loss():
            rng = random.Random(SEED)
            kept = [p for p in packets if rng.random() >= LOSS_RATE]
            capture = work / f"{codec}-randomloss.rtp"
            write_framed(capture, kept)
            got = frame_hashes(codec, depacketize(codec, capture, work / f"{codec}-rloss.{ext}"))
            return is_subsequence(got, reference), (
                f"{len(packets) - len(kept)} packets dropped, {len(got)}/{len(reference)} "
                "frames shown, all exact and in order")
        checks.attempt(f"{codec} random loss", random_loss)
        checks.attempt(f"{codec} loss", loss)

    for codec in sorted({"h264", "av1", "vp9"} & set(args.codecs.split(","))):
        def top_layer_lost(codec=codec):
            ext = EXTENSION[codec]
            stream = work / f"{codec}-t2.{ext}"
            capture = work / f"{codec}-t2.rtp"
            # A fixed two-layer structure (not the adaptive one), with the
            # Dependency Descriptor for H.264 too: without it an H.264
            # receiver cannot tell a lost top-layer frame from a lost
            # reference and waits for a key frame.
            dd = ["--dd-id", 1 if codec != "vp9" else "none"]
            run([nc, "encode", "--codec", codec, *encode_common, "--temporal-layers", 2,
                 "--param", "layers.adaptive=0", "--output", stream, "--rtp", capture,
                 "--rtp-dd-id", dd[1]])
            reference = frame_hashes(codec, stream)
            packets = read_framed(capture)
            # Frame 3 is a top-layer (T1) frame of the two-layer structure.
            lost = 3
            gone = set(units_by_timestamp(packets)[lost])
            damaged = work / f"{codec}-t2-loss.rtp"
            write_framed(damaged, [p for i, p in enumerate(packets) if i not in gone])
            got = frame_hashes(codec, depacketize(codec, damaged, work / f"{codec}-t2-out.{ext}", *dd))
            expected = reference[:lost] + reference[lost + 1:]
            return got == expected, f"frame {lost} (T1) lost: {len(got)}/{len(expected)} others shown"
        checks.attempt(f"{codec} top layer lost", top_layer_lost)

    if "av1" in args.codecs.split(","):
        for mode, points in SVC_CASES:
            stream = work / f"svc-{mode}.ivf"
            capture = work / f"svc-{mode}.rtp"
            run([nc, "encode", "--codec", "av1", *encode_common, "--scalability", mode,
                 "--output", stream, "--rtp", capture])
            for op, spatial, temporal in points:
                def forwarded(op=op, spatial=spatial, temporal=temporal, stream=stream,
                              capture=capture, mode=mode):
                    expected = work / f"svc-{mode}-op{op}.ivf"
                    run([nc, "extract", "--codec", "av1", "--input", stream,
                         "--operating-point", op, "--output", expected])
                    got = depacketize("av1", capture, work / f"svc-{mode}-S{spatial}T{temporal}.ivf",
                                      "--layers", f"{spatial}:{temporal}")
                    a, b = frame_hashes("av1", expected), frame_hashes("av1", got)
                    sizes = (Path(got).stat().st_size, Path(expected).stat().st_size)
                    return a == b and len(a) > 0, (f"op {op}: {len(b)} frames identical to extract; "
                                                   f"{sizes[0]} vs {sizes[1]} bytes")
                checks.attempt(f"av1 {mode} forward S{spatial}T{temporal}", forwarded)

    failed = [row for row in checks.rows if not row[1]]
    print(f"\n{len(checks.rows) - len(failed)}/{len(checks.rows)} checks passed")
    if not args.keep:
        shutil.rmtree(work)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
