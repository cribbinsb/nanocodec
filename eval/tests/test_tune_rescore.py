"""Retained V1 scoring inherits gates by exact bytes, never by encoder reruns."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import tune, tune_rescore as rescore
from nceval.codecs import Codec


class RetainedRescore(unittest.TestCase):
    def fixture(self, root):
        work = root / "screen"
        (work / "retained").mkdir(parents=True)
        (work / "curve-checkpoints").mkdir()
        binary = root / "encoder"
        binary.write_bytes(b"frozen encoder")
        source = root / "source.yuv"
        source.write_bytes(b"source")
        clean = root / "clean.yuv"
        clean.write_bytes(b"clean")
        clip = tune.ClipRef(
            "one",
            source,
            256,
            256,
            30,
            2,
            "synthetic_noise",
            10000,
            clean_reference=clean,
        )
        identity = {
            "binary": rescore._sha(binary),
            "protocol": "frozen screening protocol",
        }
        reference = tune.replace(clip, yuv=clean)
        cfp = tune.clip_fingerprint(clip)
        rfp = tune.clip_fingerprint(reference)
        key = {
            "evaluation": identity,
            "clip": cfp,
            "reference": rfp,
            "overrides": {"hevc.pre_transform_zero_scale": 2},
            "frames": 2,
            "quantizers": [1, 2, 3, 4],
            "metric_profile": "psnr-ms-ssim",
            "keep_streams": True,
            "extend_noise_range": True,
        }
        raw = bytes([77]) * (clip.fmt.frame_size * 2)
        reconsha = hashlib.sha256(raw).hexdigest()
        rates = []
        for q in [1, 2, 3, 4]:
            stream = work / "retained" / f"one_q{q}.hevc"
            stream.write_bytes(bytes([q]) * 10)
            rate = 10 * 8 / 1000 / (2 / clip.fps)
            rates.append(rate)
            m = {
                "binary_sha256": identity["binary"],
                "overrides": key["overrides"],
                "clip": "one",
                "clip_fingerprint": cfp,
                "reference_fingerprint": rfp,
                "metric_profile": "psnr-ms-ssim",
                "frames": 2,
                "width": 256,
                "height": 256,
                "fps": 30,
                "quantizer": q,
                "intra_period": 300,
                "conformance": "two decoders and encoder reconstruction passed",
                "stream_sha256": rescore._sha(stream),
                "recon_sha256": reconsha,
                "kbps": rate,
                "source_cap_kbps": clip.max_test_kbps,
                "psnr_yuv": 29 + q,
                "ms_ssim": 0.89 + q * 0.01,
            }
            stream.with_suffix(".hevc.json").write_text(json.dumps(m))
        curve = tune.Curve(
            rates,
            [30, 31, 32, 33],
            quantizers=[1, 2, 3, 4],
            ms_ssim=[0.9, 0.91, 0.92, 0.93],
        )
        (work / "curve-checkpoints/one.json").write_text(
            json.dumps(
                {"identity": key, "curve": tune.curve_record(curve, "psnr-ms-ssim")}
            )
        )
        return work, binary, clip, identity, raw

    def patches(self, clip, identity):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch.object(tune, "load_clips", return_value=[clip]))
        stack.enter_context(
            patch.object(tune, "evaluation_identity", return_value=identity)
        )
        stack.enter_context(patch.object(tune, "PROXY_FRAMES", 2))
        stack.enter_context(patch.object(tune, "payload_bytes", return_value=10))
        return stack

    def test_rescores_once_and_resumes_without_decode_metric_or_encode(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            work, binary, clip, identity, raw = self.fixture(root)
            out = root / "v1/result.json"
            paths = SimpleNamespace(work=root / "scratch")
            before = {
                str(p.relative_to(work)): rescore._sha(p)
                for p in work.rglob("*")
                if p.is_file()
            }

            def decode(_d, _s, yuv, _c, _f, _p):
                yuv.write_bytes(raw)
                return ""

            score = SimpleNamespace(
                summary={"vmaf_v1": 61.5}, per_frame=[{"vmaf_v1": 61}, {"vmaf_v1": 62}]
            )
            with self.patches(clip, identity), patch.object(
                rescore.conformance, "decode_to_file", side_effect=decode
            ) as dec, patch.object(
                rescore.metrics,
                "vmaf_v1_protocol",
                return_value={"model": "pinned", "subsample": 1},
            ), patch.object(
                rescore.metrics, "vmaf_v1", return_value=score
            ) as metric, patch.object(
                rescore, "check_disk"
            ), patch.object(
                tune, "run"
            ) as encode:
                result = rescore.rescore(binary, Codec.HEVC, work, out, paths)
                self.assertTrue(result["complete"])
                self.assertEqual(dec.call_count, 4)
                self.assertEqual(metric.call_count, 4)
                encode.assert_not_called()
                self.assertEqual(result["curves"]["one"]["vmaf_v1"], [61.5] * 4)
                self.assertNotIn("ms_ssim", result["curves"]["one"])
                rescore.rescore(binary, Codec.HEVC, work, out, paths)
                self.assertEqual(dec.call_count, 4)
                self.assertEqual(metric.call_count, 4)
            self.assertEqual(
                before,
                {
                    str(p.relative_to(work)): rescore._sha(p)
                    for p in work.rglob("*")
                    if p.is_file()
                },
            )

    def test_source_reference_binary_protocol_stream_or_gate_changes_reject(self):
        for change in ["source", "reference", "binary", "protocol", "stream", "gate"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                work, binary, clip, identity, _ = self.fixture(root)
                if change == "source":
                    clip.yuv.write_bytes(b"changed source")
                if change == "reference":
                    clip.clean_reference.write_bytes(b"changed clean")
                if change == "binary":
                    identity = {**identity, "binary": "different"}
                if change == "protocol":
                    identity = {**identity, "protocol": "different"}
                if change == "stream":
                    (work / "retained/one_q1.hevc").write_bytes(b"changed stream")
                if change == "gate":
                    p = work / "retained/one_q1.hevc.json"
                    m = json.loads(p.read_text())
                    m["conformance"] = "not gated"
                    p.write_text(json.dumps(m))
                with self.patches(clip, identity):
                    with self.assertRaises(ValueError):
                        rescore.retained_points(binary, Codec.HEVC, work)

    def test_wrong_decoded_reconstruction_rejects_before_metric(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            work, binary, clip, identity, raw = self.fixture(root)
            paths = SimpleNamespace(work=root / "scratch")

            def decode(_d, _s, yuv, _c, _f, _p):
                yuv.write_bytes(bytes([78]) * len(raw))
                return ""

            with self.patches(clip, identity), patch.object(
                rescore.conformance, "decode_to_file", side_effect=decode
            ), patch.object(
                rescore.metrics, "vmaf_v1_protocol", return_value={}
            ), patch.object(
                rescore.metrics, "vmaf_v1"
            ) as metric, patch.object(
                rescore, "check_disk"
            ):
                with self.assertRaisesRegex(ValueError, "reconstruction hash"):
                    rescore.rescore(
                        binary, Codec.HEVC, work, root / "output.json", paths
                    )
                metric.assert_not_called()

    def test_output_cannot_modify_screening_work_and_v1_protocol_invalidates_cache(
        self,
    ):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            work, binary, clip, identity, _ = self.fixture(root)
            out = root / "output.json"
            with self.assertRaises(ValueError):
                rescore.rescore(binary, Codec.HEVC, work, work / "output.json")
            with self.patches(clip, identity), patch.object(
                rescore.metrics, "vmaf_v1_protocol", return_value={"new": "protocol"}
            ):
                out.write_text(
                    json.dumps(
                        {"identity": {"old": "protocol"}, "points": {}, "curves": {}}
                    )
                )
                with self.assertRaisesRegex(ValueError, "cache identity"):
                    rescore.rescore(
                        binary,
                        Codec.HEVC,
                        work,
                        out,
                        SimpleNamespace(work=root / "scratch"),
                    )


if __name__ == "__main__":
    unittest.main()
