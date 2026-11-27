"""Score the demo's synthetic scenes (web/scene.js) with every codec at the demo's CBR settings.

Sources are rendered by tools/dev/scene_render.py (missing ones are rendered first). Each
point is a cbr.Bench point (encoder reconstruction, PSNR per plane, VMAF v1 and MS-SSIM
on every frame, per-frame bits and key flags; no decode), at the demo's live settings
(web/panel.js, web/options.js cliCommand): CBR target, peak 2x target, 1 s buffer, one
key per 30 s, the preset, still-camera off, no overrides.

  python3 tools/dev/scene_sanity.py --bin BIN --sources DIR --work DIR --out JSON \
      --sizes 854x480,1280x720 --scenes crowd,... --codecs h264,... --presets medium \
      --rates 200,400,800 --frames 300 --workers 8
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval.cbr import Bench, Family, Point  # noqa: E402
from nceval.clips import Clip  # noqa: E402
from nceval.codecs import Codec  # noqa: E402
from nceval.config import load_paths  # noqa: E402
from nceval.gatepolicy import PointCheck  # noqa: E402

FPS = 30
KEY_SECONDS = 30
PEAK_TO_TARGET = 2  # web/view-settings.js PEAK_TO_TARGET
# The scenes are drawn, not decoded: no source-bitrate limit applies.
SYNTHETIC_SOURCE_KBPS = 1e6
KEEP = ("kbps", "target_kbps", "rate_error", "psnr_yuv", "vmaf_v1", "ms_ssim", "worst_psnr_yuv",
        "worst_vmaf_v1", "overruns", "max_fullness", "frame_bits", "frame_psnr_yuv",
        "frame_vmaf_v1", "frame_psnr_y", "frame_key")


def main():
    ap = argparse.ArgumentParser()
    for name in ("bin", "sources", "work", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--sizes", default="854x480,1280x720")
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--codecs", default="h264,hevc,vvc,av1,vp9,av2")
    ap.add_argument("--presets", default="medium")
    ap.add_argument("--rates", default="200,400,800")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    # A scene's sceneParams version when it is not 1 (scene=version,...): part of the clip
    # name, so the point cache (keyed on name and size) never serves another version's points.
    ap.add_argument("--versions", default="")
    # Parameter overrides for an A/B (name=value,...), passed as --param and named in the family.
    ap.add_argument("--params", default="")
    ap.add_argument("--label", default="", help="family name suffix for the --params variant")
    args = ap.parse_args()
    sources, frames = Path(args.sources), args.frames
    versions = dict(v.split("=") for v in args.versions.split(",") if v)
    clips = []
    for size in args.sizes.split(","):
        w, h = map(int, size.split("x"))
        missing = [s for s in args.scenes.split(",") if not (sources / f"{s}_{size}_{frames}f.yuv").exists()]
        if missing:
            subprocess.run([sys.executable, str(Path(__file__).with_name("scene_render.py")),
                            str(sources), size, str(frames), *missing], check=True)
        for scene in args.scenes.split(","):
            tag = f"-v{versions[scene]}" if scene in versions else ""
            clips.append(Clip(name=f"scene-{scene}{tag}_{size}_{frames}f", yuv=sources / f"{scene}_{size}_{frames}f.yuv",
                              width=w, height=h, fps=FPS, frames=frames, source_kbps=SYNTHETIC_SOURCE_KBPS,
                              content_class="synthetic", split="train"))
    paths = load_paths()
    bench = Bench(paths, frames, KEY_SECONDS * FPS, Path(args.work), metric_set="real")
    jobs = []
    for codec in args.codecs.split(","):
        for preset in args.presets.split(","):
            for rate in map(int, args.rates.split(",")):
                overrides = tuple(x for p in args.params.split(",") if p for x in ("--param", p))
                family = Family(f"nanocodec-{codec}-{preset}-peak{PEAK_TO_TARGET * rate}{args.label}", Codec(codec),
                                Path(args.bin), ("--preset", preset, "--peak-bitrate", str(PEAK_TO_TARGET * rate),
                                                 *overrides), own=True)
                jobs += [(codec, preset, rate, Point(family, clip, kbps=rate, check=PointCheck.NONE)) for clip in clips]

    def run(job):
        codec, preset, rate, point = job
        try:
            r = bench.measure(point)
        except Exception as error:  # report, keep going
            print(f"FAIL {codec} {preset} {rate} {point.clip.name}: {error}", flush=True)
            return None
        print(f"{point.clip.name} {codec} {preset} {rate}: {r['kbps']:.0f} kbps "
              f"{r['psnr_yuv']:.2f} dB vmaf {r['vmaf_v1']:.1f}", flush=True)
        return {"scene": point.clip.name, "codec": codec, "label": args.label, "preset": preset, "rate": rate,
                **{k: r.get(k) for k in KEEP}}

    with ThreadPoolExecutor(args.workers) as pool:
        results = [r for r in pool.map(run, jobs) if r]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results))
    print(f"{len(results)}/{len(jobs)} points -> {args.out}")


if __name__ == "__main__":
    main()
