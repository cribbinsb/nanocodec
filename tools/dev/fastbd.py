"""Quick BD-rate (PSNR-YUV) of a nanocodec build vs anchor points already in
the main results DB, from our encoder's own reconstruction, quick clips,
10 s IP. Points are not decoded unless FASTBD_GATE asks
(nceval.gatepolicy: conformance is checked when a change lands). Points are
streamed (nceval.point): the reconstruction is scored while it encodes and
no video file is written.

Both PSNR-YUV and VMAF-NEG come from the same encodes (one run, two tables).
Cost knobs (environment): FASTBD_FRAMES caps frames per clip (default: whole
clip; 150 keeps P frames dominant at 10 s IP), FASTBD_WORKERS (default 6),
FASTBD_VMAF_SUBSAMPLE scores every Nth frame for VMAF (default 3), FASTBD_QPS
overrides the quantiser grid, FASTBD_GATE=off|sample|full (default off) which
decoders check the points (nceval.gatepolicy; our own decoder is BINARY).
Each point records its gate level.
FASTBD_JSON saves the measured points for later comparison. ANCHOR may be
that JSON file to compare two builds over matching clip lengths instead
of comparing shortened encodes against full-clip database anchors.

usage: python3 -I fastbd.py CODEC ANCHOR BINARY [ENV=VAL | name=value ...]
  ENV=VAL with an upper-case name sets an environment variable; a lower-case
  dotted name=value is passed as `--param name=value` (see `nanocodec params`).
  e.g. fastbd.py av1 svtav1-m12 targets/av1/release/nanocodec NC_FOO=1
"""
import hashlib, json, os, sqlite3, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval.bdrate import bd_rate
from nceval.codecs import Codec
from nceval.diskguard import check_disk
from nceval.config import load_paths
from nceval.clips import load_clipset
from nceval.fingerprint import clip_fingerprint, measurement_fingerprint
from nceval.sweep import payload_bytes
from nceval.metrics import Measurement, NativeVmafMetric, PsnrMetric, StreamSpec
from nceval.gatepolicy import (EVALUATION_GATE, GatePolicy, combination_key, file_sha256,
                               gate_arguments, point_check)
from nceval.point import run_point
from nceval.yuv import YuvFormat

codec, anchor, binary = sys.argv[1:4]
settings = [a.split("=", 1) for a in sys.argv[4:]]
env = dict(os.environ, **{k: v for k, v in settings if k.isupper()})
param_args = [x for k, v in settings if not k.isupper() for x in ("--param", f"{k}={v}")]
SUFFIX = {"h264": ".264", "hevc": ".hevc", "av1": ".ivf"}
QUANTIZERS = {"h264": [22, 26, 30, 34, 38, 42, 46, 50], "hevc": [22, 26, 30, 34, 38, 42, 46, 50],
              "av1": [40, 65, 90, 115, 140, 165, 190, 215, 240, 255]}
PATHS = load_paths()
clips = {c["name"]: c for c in json.load(open(PATHS.data_root / "clips/quick.json"))["clips"]}
names = ["static_chirla_s000c3", "crowd_mot17_09", "pan_jaad_0117", "bodycam_airport", "lowlight_pp22_004"]
# FASTBD_QPS="40,42,..." overrides the grid, e.g. to keep four legal points
# under a low source bitrate cap.
qps = ([int(q) for q in os.environ["FASTBD_QPS"].split(",")] if os.environ.get("FASTBD_QPS")
       else QUANTIZERS[codec])
FRAMES = int(os.environ.get("FASTBD_FRAMES", "0"))
WORKERS = int(os.environ.get("FASTBD_WORKERS", "6"))
VMAF_SUBSAMPLE = int(os.environ.get("FASTBD_VMAF_SUBSAMPLE", "3"))
GATE = GatePolicy(os.environ.get("FASTBD_GATE", EVALUATION_GATE.value))
if not 1 <= WORKERS <= 8:
    raise ValueError("FASTBD_WORKERS must be between 1 and 8")
clip_specs = {c.name: c for c in load_clipset(
    PATHS.data_root / "clips/quick.json", PATHS.data_root).clips}
# Bump when what is measured changes (grid, metric definitions, sampling);
# plumbing changes that leave every number identical keep saved anchors valid.
PROTOCOL_REVISION = "fastbd-2"
identity = {"clips": {n: clip_fingerprint(clip_specs[n]) for n in names},
            "measurement": measurement_fingerprint(),
            "protocol": PROTOCOL_REVISION}
db = sqlite3.connect(f"file:{PATHS.results_db}?mode=ro", uri=True)
saved_anchor = None
if Path(anchor).is_file():
    saved_anchor = json.loads(Path(anchor).read_text())
    if saved_anchor.get("identity") != identity:
        raise ValueError("saved anchor has a different rate protocol or clip identity; regenerate it")
    if (saved_anchor["codec"], saved_anchor["frames"], saved_anchor["vmaf_subsample"]) != (
            codec, FRAMES, VMAF_SUBSAMPLE):
        raise ValueError("saved anchor must use the same codec, frame count and VMAF subsampling")
check_disk(PATHS.data_root / "work")
tmp = Path(tempfile.mkdtemp(dir=PATHS.data_root / "work"))


def vmaf_neg_metric(spec: StreamSpec) -> NativeVmafMetric:
    """Mean VMAF-NEG over every VMAF_SUBSAMPLE-th frame (one libvmaf thread)."""
    model = PATHS.anchors / "share" / "vmaf" / "model" / "vmaf_v0.6.1neg.json"
    return NativeVmafMetric(
        spec, "vmaf_neg", ["--subsample", str(VMAF_SUBSAMPLE),
                           "--model", f"path={model}:name=vmaf_neg", "--threads", "1"],
        lambda report: Measurement({"vmaf_neg": report["pooled_metrics"]["vmaf_neg"]["mean"]}))


def run(args):
    check_disk(PATHS.data_root / "work")
    name, qp = args
    c = clips[name]
    frames = min(c["frames"], FRAMES) if FRAMES else c["frames"]
    out = tmp / f"{name}_{qp}{SUFFIX[codec]}"
    ip = round(c["fps"] * 10)
    yuv = str(PATHS.data_root / c["yuv"])
    command = lambda recon: [
        binary, "encode", "--codec", codec, "--input", yuv, "--width", str(c["width"]),
        "--height", str(c["height"]), "--fps", str(c["fps"]), "--qp", str(qp),
        "--frames", str(frames), "--intra-period", str(ip), "--output", str(out),
        "--display-recon", recon, *param_args]
    spec = StreamSpec(Path(yuv), YuvFormat(c["width"], c["height"]), frames, c["fps"], PATHS)
    combination = combination_key(file_sha256(Path(binary)), dict(settings), name)
    check = point_check(GATE, combination, qps.index(qp), len(qps))
    try:
        point = run_point(command, out, Codec(codec), spec, PATHS,
                          [PsnrMetric(spec), vmaf_neg_metric(spec)],
                          env=env, **gate_arguments(check, Path(binary)))
        if not point.gate.passed:
            raise RuntimeError(f"{name} qp={qp}: conformance failed: {point.gate.reason}")
        kbps = payload_bytes(out, Codec(codec), frames) * 8 / 1000 / (frames / c["fps"])
    finally:
        out.unlink(missing_ok=True)
    if not clip_specs[name].allows_kbps(kbps):
        return None  # Never scored above the source cap.
    summary = point.measurement.summary
    return name, kbps, {"psnr_yuv": summary["psnr_yuv"], "vmaf_neg": summary["vmaf_neg"],
                        "gate_level": point.gate.level}


with ThreadPoolExecutor(WORKERS) as ex:
    res = [point for point in ex.map(run, [(n, q) for n in names for q in qps])
           if point is not None]
tmp.rmdir()
levels = [q["gate_level"] for _, _, q in res]
print(f"gate ({GATE.value}): " + ", ".join(f"{level} {levels.count(level)}"
                                           for level in sorted(set(levels))))
if os.environ.get("FASTBD_JSON"):
    Path(os.environ["FASTBD_JSON"]).write_text(json.dumps({
        "codec": codec, "frames": FRAMES, "vmaf_subsample": VMAF_SUBSAMPLE, "gate": GATE.value,
        "binary": binary, "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "settings": settings, "points": res, "identity": identity,
    }, indent=2))
# Anchor points come from full-length clips; with FASTBD_FRAMES the comparison
# is approximate (shorter clip, same anchor curve) and is labelled as such.
note = f" (first {FRAMES} frames vs full-clip anchor: approximate)" if FRAMES and saved_anchor is None else ""
for metric in ("psnr_yuv", "vmaf_neg"):
    print(f"== {codec} vs {anchor}, {metric}{note}")
    total = []
    for n in names:
        ours = sorted((k, q[metric]) for nn, k, q in res if nn == n)
        if saved_anchor is not None:
            anc = sorted((k, q[metric]) for nn, k, q in saved_anchor["points"] if nn == n)
        else:
            anc = db.execute(
                "select e.kbps, m.value from encodes e join metrics m on m.encode_id=e.id where e.encoder=? "
                "and e.structure='ip10s-tl1' and e.clip=? and e.conformance='pass' and m.name=? order by e.kbps",
                (anchor, n, metric)).fetchall()
        anc = [(r, q) for r, q in anc if clip_specs[n].allows_kbps(r)]
        if len(anc) < 4 or len(ours) < 4:
            raise ValueError(f"{n}: need four legal points per curve under the source bitrate cap")
        r = bd_rate([a[0] for a in anc], [a[1] for a in anc], [o[0] for o in ours], [o[1] for o in ours])
        if r.bd_rate_percent is None or not r.reliable:
            raise ValueError(f"{n}: insufficient quality overlap for {metric}")
        total.append(r.bd_rate_percent)
        print(f"{n:24s} {r.bd_rate_percent:+7.1f}%  overlap {r.overlap_fraction:.2f}")
    if total:
        print(f"{'mean':24s} {sum(total)/len(total):+7.1f}%")
