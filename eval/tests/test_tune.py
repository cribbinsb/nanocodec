"""Tuning must not hide unmeasurable clips or reuse anchors after a rebuild."""

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import tune
from nceval.conformance import GateResult
from fake_point import FakeEncodePoint


class FixedGridMeasurements(unittest.TestCase):
    def setUp(self):
        identity = patch.object(tune, "evaluation_identity", return_value={"binary": "frozen"})
        fingerprint = patch.object(tune, "clip_fingerprint",
            side_effect=lambda clip: str(getattr(clip, "yuv", clip.name)))
        self.identity = identity.start()
        fingerprint.start()
        self.addCleanup(identity.stop)
        self.addCleanup(fingerprint.stop)

    def jobs(self, count=4):
        return [tune.FixedGridJob(SimpleNamespace(name=f"clip{i}", frames=30, clean_reference=None),
                                  12 + i, (100 + i, 200 + i)) for i in range(count)]

    def test_parallel_bounds_isolation_and_explicit_requests(self):
        jobs = self.jobs()
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = maximum = 0
        seen = []
        def measure(binary, codec, clip, frames, overrides, work, quantizers, **options):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                seen.append((clip.name, frames, quantizers, work, options))
            barrier.wait(timeout=5)
            (work / "same-file").write_text(clip.name)
            with lock:
                active -= 1
            return tune.Curve([100], [40], vmaf_v1=[80], quantizers=[quantizers[0]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(tune, "measure_curve", side_effect=measure):
            root = Path(tmp)
            decoder = root / "own"
            decoder.write_bytes(b"decoder")
            result = tune.measure_fixed_grid(Path("/encoder"), tune.Codec.AV1, jobs,
                {}, root, workers=2, own_decoder=decoder)
            self.assertEqual(list(result), [j.clip.name for j in jobs])
            self.assertEqual(maximum, 2)
            self.assertEqual(len({row[3] for row in seen}), 4)
            for name, frames, quantizers, work, options in seen:
                job = next(j for j in jobs if j.clip.name == name)
                self.assertEqual((frames, quantizers), (job.frames, list(job.quantizers)))
                self.assertEqual(options, dict(fixed_grid=True,
                    metric_profile=tune.V1_METRIC_PROFILE, keep_streams=True,
                    own_decoder=decoder))
                self.assertEqual((work / "same-file").read_text(), name)
                checkpoint = json.loads((work / "curve.json").read_text())
                self.assertEqual(checkpoint["clip"], name)
                self.assertEqual(checkpoint["identity"], {
                    "evaluation": {"binary": "frozen"}, "codec": "av1", "overrides": {},
                    "clip": name, "reference": name, "frames": frames,
                    "quantizers": quantizers, "keep_streams": True,
                    "own_decoder_sha256": tune.hashlib.sha256(b"decoder").hexdigest()})
            self.identity.assert_called_once_with(Path("/encoder"), tune.V1_METRIC_PROFILE)

    def test_checkpoint_distinguishes_input_and_clean_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = tune.ClipRef("noise", root / "noisy", 4, 4, 30, 1,
                                "synthetic_noise", 10000, clean_reference=root / "clean")
            curve = tune.Curve([100], [40], vmaf_v1=[80], quantizers=[100])
            with patch.object(tune, "measure_curve", return_value=curve):
                tune.measure_fixed_grid(Path("/encoder"), tune.Codec.AV1,
                    [tune.FixedGridJob(clip, 1, (100,))], {"x": 0.5}, root)
            identity = json.loads((root / "clip-0000/curve.json").read_text())["identity"]
            self.assertEqual(identity["clip"], str(root / "noisy"))
            self.assertEqual(identity["reference"], str(root / "clean"))
            self.assertEqual(identity["overrides"], {"x": 0.5})

    def test_failure_propagates_without_starting_remaining_clips(self):
        started = []
        other_started = threading.Event()
        def measure(binary, codec, clip, *args, **kwargs):
            started.append(clip.name)
            if clip.name == "clip0":
                raise RuntimeError("encoder failed")
            other_started.set()
            return tune.Curve([100], [40], vmaf_v1=[80], quantizers=[100])
        with tempfile.TemporaryDirectory() as tmp, patch.object(tune, "measure_curve", side_effect=measure):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                tune.measure_fixed_grid(Path("/encoder"), tune.Codec.AV1,
                    self.jobs(), {}, Path(tmp), workers=1)
        self.assertEqual(started, ["clip0"])

    def test_invalid_jobs_fail_before_measurement(self):
        for workers in [0, 9, True, 1.5]:
            with patch.object(tune, "measure_curve") as measure:
                with self.assertRaises(ValueError):
                    tune.measure_fixed_grid(Path("/encoder"), tune.Codec.AV1,
                        self.jobs(), {}, Path("/unused"), workers=workers)
                measure.assert_not_called()
        for job in [tune.FixedGridJob(SimpleNamespace(name="../escape", frames=30), 10, (100,)),
                    tune.FixedGridJob(SimpleNamespace(name="x", frames=30), 31, (100,)),
                    tune.FixedGridJob(SimpleNamespace(name="x", frames=30), 10, (100, 100)),
                    tune.FixedGridJob(SimpleNamespace(name="x", frames=30), 10, ())]:
            with self.assertRaises(ValueError):
                tune.measure_fixed_grid(Path("/encoder"), tune.Codec.AV1,
                    [job], {}, Path("/unused"))

    def test_measure_curve_gate_failure_cannot_reach_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = tune.ClipRef("clip", root / "source.yuv", 4, 4, 30, 1, "static", 10000)
            failed = tune.PointResult(GateResult(False, "nanocodec and ffmpeg differ at frame 0"),
                                      "recon", None)

            def point(command, stream, *args, **kwargs):
                stream.write_bytes(b"stream")
                return failed
            with patch.object(tune, "check_disk"), patch.object(tune, "load_paths"), \
                    patch.object(tune, "file_sha256", return_value="binary"), \
                    patch.object(tune, "run_point", side_effect=point) as run_point:
                with self.assertRaisesRegex(RuntimeError, "conformance failed.*differ"):
                    tune.measure_curve(Path("/encoder"), tune.Codec.AV1, clip, 1, {},
                        root, [100], fixed_grid=True, own_decoder=Path("/own"),
                        gate=tune.GatePolicy.FULL)
            self.assertEqual(run_point.call_args.args[-1], Path("/own"))
            self.assertEqual(list(root.iterdir()), [])


class TuningGuards(unittest.TestCase):
    def test_trimmed_reference_repairs_partial_cache_and_reuses_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.yuv'
            source.write_bytes(bytes(range(96)))
            clip = SimpleNamespace(name='shared', yuv=source,
                                   fmt=SimpleNamespace(frame_size=24))
            cached = root / 'refs/shared_2f_fixed.yuv'
            cached.parent.mkdir()
            cached.write_bytes(b'partial')
            with patch.object(tune, 'TUNING_DIR', root), \
                    patch.object(tune, 'clip_fingerprint', return_value='fixed'):
                self.assertEqual(tune.trimmed_reference(clip, 2), cached)
                self.assertEqual(cached.read_bytes(), bytes(range(48)))
                source.unlink()  # A valid cached reference needs no source read.
                self.assertEqual(tune.trimmed_reference(clip, 2), cached)
            self.assertEqual(list(cached.parent.glob('*.tmp')), [])

    def test_trimmed_reference_does_not_publish_short_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.yuv'
            source.write_bytes(b'short')
            clip = SimpleNamespace(name='shared', yuv=source,
                                   fmt=SimpleNamespace(frame_size=24))
            with patch.object(tune, 'TUNING_DIR', root), \
                    patch.object(tune, 'clip_fingerprint', return_value='fixed'):
                with self.assertRaisesRegex(ValueError, 'shorter than 2 complete frames'):
                    tune.trimmed_reference(clip, 2)
            self.assertEqual(list((root / 'refs').iterdir()), [])

    def test_trimmed_reference_concurrent_reader_never_reuses_unpublished_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.yuv'
            source.write_bytes(bytes(range(96)))
            clip = SimpleNamespace(name='shared', yuv=source,
                                   fmt=SimpleNamespace(frame_size=24))
            cached = root / 'refs/shared_2f_fixed.yuv'
            ready, release = threading.Event(), threading.Event()
            lock = threading.Lock()
            first = True
            real_replace = Path.replace

            def delay_first_publication(temporary, destination):
                nonlocal first
                with lock:
                    pause, first = first, False
                if pause:
                    ready.set()
                    if not release.wait(5):
                        raise RuntimeError('test publication barrier timed out')
                return real_replace(temporary, destination)

            with patch.object(tune, 'TUNING_DIR', root), \
                    patch.object(tune, 'clip_fingerprint', return_value='fixed'), \
                    patch.object(Path, 'replace', autospec=True,
                                 side_effect=delay_first_publication), \
                    ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(tune.trimmed_reference, clip, 2)
                try:
                    self.assertTrue(ready.wait(5))
                    self.assertFalse(cached.exists())
                    # The second caller can publish its own complete copy. It
                    # cannot return an empty/partial file owned by the writer.
                    reader = pool.submit(tune.trimmed_reference, clip, 2)
                    self.assertEqual(reader.result(timeout=5).read_bytes(), bytes(range(48)))
                finally:
                    release.set()
                self.assertEqual(writer.result(timeout=5), cached)
            self.assertEqual(cached.read_bytes(), bytes(range(48)))
            self.assertEqual(list(cached.parent.glob('*.tmp')), [])

    def test_front_filters_constraints_before_dominance(self):
        def trial(number, values, constraint=(0, 0), state=tune.optuna.trial.TrialState.COMPLETE):
            return SimpleNamespace(number=number, values=values, state=state,
                                   user_attrs={'constraint': constraint})

        trials = [trial(0, [-5, -3, 0.9]),
                  trial(1, [-10, -8, 0.8], constraint=(1, -2)),
                  trial(2, [-4, -2, 1.0]),
                  trial(3, [-8, -1, 1.0]),
                  trial(4, None, state=tune.optuna.trial.TrialState.RUNNING),
                  trial(5, [-5, -3, 0.9])]
        self.assertEqual([t.number for t in tune.feasible_front(trials)], [0, 3, 5])

    def test_noise_uses_best_and_fixed_quality_without_bd(self):
        clip = tune.ClipRef("noise", Path("/noise"), 1280, 720, 30, 150,
                            "synthetic_noise", 2000, clean_reference=Path("/clean"))
        # Nonmonotonic quality is deliberate: noise must never require BD overlap.
        anchor = tune.Curve([100, 200, 400, 800], [35, 37, 36, 34], [60, 70, 65, 55])
        better = tune.Curve([800, 400, 200, 100], [35, 37, 38, 36], [56, 66, 71, 61])
        for metric in ("psnr", "vmaf_neg"):
            result = tune.noise_deltas({"noise": better}, {"noise": anchor}, [clip], metric)
            self.assertEqual(result["noise"], {"best": 1, "at_200_kbps": 1, "at_400_kbps": 1})
            self.assertEqual(tune.noise_deltas({"noise": anchor}, {"noise": anchor},
                                              [clip], metric)["noise"]["best"], 0)
        short = tune.Curve([300, 400, 600, 800], [40, 41, 42, 43], [80, 81, 82, 83])
        with self.assertRaisesRegex(ValueError, "noise.*cover 200"):
            tune.noise_deltas({"noise": short}, {"noise": anchor}, [clip], "psnr")

    def test_noise_fixed_rates_match_official_sublinear_scaling(self):
        clip = tune.ClipRef("noise", Path("/noise"), 1280, 720, 15, 150,
                            "synthetic_noise", 2000, clean_reference=Path("/clean"))
        curve = tune.Curve([100, 200, 400, 800], [35, 37, 36, 34], [60, 70, 65, 55])
        with patch.object(tune, "_interpolate_log_rate", return_value=40) as interpolate:
            tune.noise_deltas({"noise": curve}, {"noise": curve}, [clip], "psnr")
        targets = [call.args[1] for call in interpolate.call_args_list]
        self.assertEqual(targets, [200 * 0.5 ** 0.75] * 2 + [400 * 0.5 ** 0.75] * 2)

    def _check_noise_search(self, strict, cost_gop=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "nanocodec"
            binary.write_bytes(b"encoder")
            clips = [tune.ClipRef(n, root / n, 1280, 720, 30, 150, "static", 2000)
                     for n in tune.COST_CLIPS]
            clips += [tune.ClipRef("noise", root / "noise", 1280, 720, 30, 150,
                                  "synthetic_noise", 2000, clean_reference=root / "clean")]
            curve = tune.Curve([100, 200, 400, 800], [30, 33, 36, 40], quantizers=[1, 2, 3, 4], vmaf_v1=[60, 65, 70, 80])
            noisy = tune.Curve([100, 200, 400, 800], [37, 36, 35, 34], quantizers=[1, 2, 3, 4], vmaf_v1=[70, 69, 68, 67])
            worse = tune.Curve([100, 200, 400, 800], [36, 35, 34, 33], quantizers=[1, 2, 3, 4], vmaf_v1=[68, 67, 66, 65])
            base = {c.name: curve for c in clips} | {"noise": noisy}
            worse.psnr = [np.float64(v) for v in worse.psnr]
            worse.vmaf_v1 = [np.float64(v) for v in worse.vmaf_v1]
            candidate = base | {"noise": worse}
            study = tune.optuna.create_study(directions=["minimize"] * 3,
                                             storage=f"sqlite:///{root / 'study.sqlite'}")
            args = argparse.Namespace(codec="av1", binary=str(binary), params="test.value",
                                      study="noise-test", seed=1, trials=1, workers=1,
                                      noise_guard=strict, noise_report=True, cost_gop=cost_gop)
            with patch.object(tune, "TUNING_DIR", root), \
                 patch.object(tune, "specs_of", return_value={"test.value": (1, 0, 2)}), \
                 patch.object(tune, "proxy_clips", return_value=clips), \
                 patch.object(tune, "clip_fingerprint", return_value="fingerprint"), \
                 patch.object(tune, "evaluate", side_effect=[tune.Evaluation(base, 1, 1),
                                                            tune.Evaluation(candidate, 1, 4)]), \
                 patch.object(tune.optuna, "create_study", return_value=study), \
                 patch.object(tune, "print_front"):
                tune.search(args)
            trial = study.trials[0]
            self.assertEqual(trial.state, tune.optuna.trial.TrialState.COMPLETE)
            self.assertAlmostEqual(trial.values[0], 0)
            self.assertAlmostEqual(trial.values[1], 0)
            self.assertAlmostEqual(trial.values[2], 1.01 if cost_gop == 300 else 1)
            self.assertEqual(study.user_attrs["cost_gop_period"], cost_gop)
            self.assertEqual(trial.user_attrs["instruction_cost"]["key_million"], 4)
            self.assertAlmostEqual(trial.user_attrs["instruction_cost"]["ip300_ratio"], 1.01)
            self.assertEqual(trial.user_attrs["constraint"], [-3, -3, 1, 2] if strict else [-3, -3])
            self.assertEqual(trial.user_attrs["noise_no_loss_constraint"], [1, 2])
            self.assertFalse(trial.user_attrs["noise_no_loss_passed"])
            self.assertIn("noise_psnr_deltas", trial.user_attrs)
            self.assertIn("noise_vmaf_v1_deltas", trial.user_attrs)
            self.assertEqual(tune.feasible_front(study.trials), [] if strict else [trial])

    def test_noise_loss_is_a_constraint_not_a_bd_objective(self):
        self._check_noise_search(True)

    def test_noise_report_keeps_losses_visible_without_rejecting_pareto_trial(self):
        self._check_noise_search(False)

    def test_gop300_objective_prices_key_cost_while_p_default_is_unchanged(self):
        self._check_noise_search(False, cost_gop=300)
        base = tune.InstructionCost(1, 4)
        self.assertEqual(tune.cpu_objective_cost(base, 0), 1)
        self.assertEqual(tune.cpu_objective_cost(base, 1), 4)
        self.assertAlmostEqual(tune.cpu_objective_cost(base, 300), 1.01)
        with self.assertRaises(ValueError):
            tune.cpu_objective_cost(base, -1)

    def test_gop_objective_report_uses_correct_label_and_keeps_cost_details(self):
        trial = SimpleNamespace(number=0, values=[0, 0, 1.01], params={},
                                state=tune.optuna.trial.TrialState.COMPLETE,
                                user_attrs={"constraint": [-3, -3], "instruction_cost": {
                                    "p_million": 1, "key_million": 4, "ip300_ratio": 1.01}})
        study = SimpleNamespace(study_name="gop", trials=[trial], user_attrs={"cost_gop_period": 300})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            tune.print_front(study, details=True)
        text = output.getvalue()
        self.assertIn("GOP300-instr x1.010", text)
        self.assertNotIn("P-instr", text)
        self.assertIn('"key_million": 4', text)
        self.assertIn('"p_million": 1', text)

    def test_search_space_bounds_and_numeric_categories_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "space.json"
            domains = {"lambda": {"low": 0.04, "high": 0.2, "log": True},
                       "walk": {"choices": [0, 2]}}
            path.write_text(json.dumps(domains))
            space = tune.search_space_of(list(domains), {"lambda": (0.135, 0.01, 1),
                                                         "walk": (0, 0, 2)}, str(path))
            study = tune.optuna.create_study()
            for _ in range(8):
                trial = study.ask()
                params = tune.suggest_parameters(trial, space)
                self.assertIn(params["walk"], [0, 2])
                self.assertTrue(0.04 <= params["lambda"] <= 0.2)
                self.assertTrue(trial.distributions["lambda"].log)
                self.assertEqual(trial.distributions["walk"].choices, (0, 2))
                study.tell(trial, 0)

    def test_search_space_rejects_invalid_or_incomplete_domains(self):
        invalid = [{}, {"other": {"choices": [1]}}, {"x": {"choices": []}},
                   {"x": {"choices": [1, 1]}}, {"x": {"choices": [3]}},
                   {"x": {"choices": [True]}}, {"x": {"choices": [float("nan")]}},
                   {"x": {"low": 1, "high": 0}}, {"x": {"low": 0, "high": 1, "log": True}},
                   {"x": {"low": 0, "high": float("inf")}},
                   {"x": {"low": 0, "high": 1, "extra": 2}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "space.json"
            for domain in invalid:
                with self.subTest(domain=domain):
                    path.write_text(json.dumps(domain))
                    with self.assertRaises(ValueError):
                        tune.search_space_of(["x"], {"x": (1, 0, 2)}, str(path))

    def test_exported_report_only_candidate_discloses_noise_losses(self):
        trial = SimpleNamespace(values=[-1, -2, 0.95], params={"x": 1}, user_attrs={
            "constraint": [-3, -3], "noise_no_loss_passed": False,
            "class_bd_psnr": {"bodycam": -1},
            "noise_psnr_deltas": {"noise": {"best": -0.01}}})
        study = SimpleNamespace(trials=[trial], user_attrs={"noise_report": True, "noise_guard": False, "cost_gop_period": 300})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(tune, "REPO_ROOT", Path(tmp)), \
             patch.object(tune.optuna, "load_study", return_value=study):
            tune.write_presets(argparse.Namespace(study="exploratory", trials="0:tradeoff", codec="av1"))
            text = (Path(tmp) / "presets/candidates/av1-tradeoff.toml").read_text()
            self.assertIn("Strict clean-noise no-loss passed: False", text)
            self.assertIn("noise constrained in study: False", text)
            self.assertIn('"best": -0.01', text)
            self.assertIn("class_bd_psnr", text)
            self.assertIn("GOP300-instr x", text)
            self.assertIn("Validate on VAL", text)

    def test_concurrent_sqlite_acquisition_keeps_unique_ids_and_full_seed_maps(self):
        seeds = [{"energy": 10 + i, "scale": 0.5 + i / 20, "mode": i % 3}
                 for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(tune, "TUNING_DIR", Path(tmp)):
            storage = tune.study_storage("queued")
            with storage.engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql("PRAGMA journal_mode").scalar(), "wal")
                self.assertEqual(connection.exec_driver_sql("PRAGMA busy_timeout").scalar(), 60000)
            study = tune.serialize_trial_acquisition(tune.optuna.create_study(
                study_name="queued", storage=storage, directions=["minimize"] * 3))
            for params in seeds:
                study.enqueue_trial(params)
            seen = set()
            for _ in range(4):
                barrier = threading.Barrier(4)

                def acquire(_):
                    barrier.wait()
                    trial = study.ask()
                    params = {"energy": trial.suggest_float("energy", 0, 100),
                              "scale": trial.suggest_float("scale", 0.5, 1.5),
                              "mode": trial.suggest_categorical("mode", [0, 1, 2])}
                    return trial, params

                with tune.ThreadPoolExecutor(max_workers=4) as pool:
                    batch = list(pool.map(acquire, range(4)))
                self.assertEqual(len({trial.number for trial, _ in batch}), 4)
                for trial, params in batch:
                    self.assertNotIn(trial.number, seen)
                    self.assertEqual(params, seeds[trial.number])
                    seen.add(trial.number)
                    study.tell(trial, [0, 0, 1])
            self.assertEqual(seen, set(range(16)))

    def _mock_noise_measurement(self, codec, qs, rates, extend, fixed_grid=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = tune.ClipRef("noise", root / "noisy", 1280, 720, 30, 30,
                                "synthetic_noise", 2000, clean_reference=root / "clean")
            encode = FakeEncodePoint()

            def size(stream, *_):
                q = int(stream.stem.rsplit("_q", 1)[1])
                return rates[q] * 125  # 30 frames / 30 fps = one second.

            with patch.object(tune, "check_disk"), patch.object(tune, "encode_point", side_effect=encode), patch.object(tune, "file_sha256", return_value="binary"), \
                 patch.object(tune, "payload_bytes", side_effect=size):
                curve = tune.measure_curve(Path("/binary"), codec, clip, 30, {}, root, qs,
                                           extend_noise_range=extend, fixed_grid=fixed_grid)
            # Every encoded point is gated; only legal ones are scored into the curve.
            return curve, clip, encode.quantizers, len(encode.calls), len(curve.kbps)

    def test_sample_gate_standard_decodes_first_and_one_rotating_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = tune.ClipRef("clip", root / "input", 16, 16, 30, 30, "static", 2000)
            (root / "bin").write_bytes(b"encoder")
            for policy, keep, expected in [(tune.GatePolicy.SAMPLE, False, 2),
                                           (tune.GatePolicy.FULL, False, 6),
                                           (tune.GatePolicy.SAMPLE, True, 6)]:
                encode = FakeEncodePoint()
                with patch.object(tune, "check_disk"), patch.object(tune, "encode_point", side_effect=encode), \
                     patch.object(tune, "file_sha256", return_value="binary"), \
                     patch.object(tune, "clip_fingerprint", return_value="clip"), \
                     patch.object(tune, "payload_bytes", return_value=100):
                    curve = tune.measure_curve(root / "bin", tune.Codec.H264, clip, 30, {}, root,
                                               [20, 24, 28, 32, 36, 40], fixed_grid=True,
                                               gate=policy, keep_streams=keep)
                standard = [c["standard"] for c in encode.calls]
                self.assertEqual(sum(standard), expected)
                self.assertTrue(standard[0])
                self.assertTrue(all(c["own_decoder"] == root / "bin" for c in encode.calls))
                self.assertEqual(curve.gate_levels,
                                 ["full" if s else "own-decoder" for s in standard])
                self.assertEqual(tune.curve_record(curve)["gate_levels"], curve.gate_levels)

    def test_fixed_grid_does_not_expand_single_point(self):
        curve, _, seen, gated, scored = self._mock_noise_measurement(
            tune.Codec.AV1, [220], {220: 200}, False, fixed_grid=True)
        self.assertEqual(curve.quantizers, [220])
        self.assertEqual(seen, [220])
        self.assertEqual((gated, scored), (1, 1))

    def test_fixed_grid_gates_but_does_not_replace_over_cap_points(self):
        curve, _, seen, gated, scored = self._mock_noise_measurement(
            tune.Codec.AV1, [180, 220], {180: 1100, 220: 200}, False, fixed_grid=True)
        self.assertEqual(seen, [180, 220])
        self.assertEqual(curve.quantizers, [220])
        self.assertEqual((gated, scored), (2, 1))
        with self.assertRaisesRegex(ValueError, "no requested points"):
            self._mock_noise_measurement(
                tune.Codec.AV1, [180], {180: 1100}, False, fixed_grid=True)

    def test_retained_over_cap_point_has_rate_and_gates_but_no_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "encoder"
            binary.write_bytes(b"frozen encoder")
            clip = tune.ClipRef("clip", root / "input", 2, 2, 30, 30,
                                "static", 2000)

            encode = FakeEncodePoint()

            def payload(stream, *_):
                return (1100 if "q180" in stream.name else 200) * 125

            with patch.object(tune, "check_disk"), \
                 patch.object(tune, "clip_fingerprint", return_value="input fingerprint"), \
                 patch.object(tune, "encode_point", side_effect=encode), patch.object(tune, "file_sha256", return_value="binary"), \
                 patch.object(tune, "payload_bytes", side_effect=payload):
                curve = tune.measure_curve(binary, tune.Codec.AV1, clip, 30, {}, root,
                                           [180, 220], fixed_grid=True, keep_streams=True,
                                           own_decoder=binary)
            self.assertEqual(curve.quantizers, [220])
            self.assertEqual([c["own_decoder"] for c in encode.calls], [binary, binary])
            excluded = root / "excluded" / "clip_q180.ivf"
            self.assertEqual(excluded.read_bytes(), b"180")
            record = json.loads(excluded.with_suffix(".ivf.json").read_text())
            self.assertEqual((record["kbps"], record["source_cap_kbps"]), (1100, 1000))
            self.assertTrue(record["excluded"])
            self.assertEqual(record["reason"], "source_bitrate_ceiling")
            self.assertTrue(record["own_decoder_conformance"])
            self.assertEqual(record["stream_sha256"], hashlib.sha256(b"180").hexdigest())
            self.assertEqual(record["recon_sha256"], "recon-180")
            self.assertNotIn("psnr_yuv", record)
            self.assertNotIn("vmaf_v1", record)
            legal = json.loads((root / "retained" / "clip_q220.ivf.json").read_text())
            self.assertEqual((legal["psnr_yuv"], legal["vmaf_v1"]), (40, 80))
            self.assertFalse(list(root.glob("*.yuv")))

    def test_fixed_grid_rejects_implicit_or_extensible_grid_before_work(self):
        for qs, extend in [(None, False), ([], False), ([220], True)]:
            with self.subTest(qs=qs, extend=extend), patch.object(tune, "check_disk") as disk:
                with self.assertRaisesRegex(ValueError, "fixed_grid requires"):
                    tune.measure_curve(Path("/binary"), tune.Codec.AV1, None, 1, {},
                                       Path("/work"), qs, extend_noise_range=extend,
                                       fixed_grid=True)
                disk.assert_not_called()

    def test_noise_extension_preserves_original_points_and_cost_index(self):
        for codec, qs, extra in [(tune.Codec.H264, [20, 30, 40, 44], [46, 48]),
                                 (tune.Codec.AV1, [160, 180, 200, 220], [229, 238])]:
            with self.subTest(codec=codec.name):
                rates = dict(zip(qs + extra, [600, 400, 300, 220, 210, 190]))
                curve, clip, seen, gated, scored = self._mock_noise_measurement(codec, qs, rates, True)
                self.assertEqual(seen, qs + extra)
                self.assertEqual(curve.quantizers[:4], qs)
                self.assertEqual(curve.quantizers[2], qs[2])
                self.assertEqual((gated, scored), (6, 6))
                self.assertEqual(tune.noise_range_status(curve, clip, codec), "covered")

    def test_noise_extension_stops_at_true_maximum_qp_floor(self):
        qs, extra = [20, 30, 40, 44], [46, 48, 50, 51]
        rates = dict(zip(qs + extra, [600, 400, 300, 220, 215, 210, 208, 205]))
        curve, clip, seen, gated, scored = self._mock_noise_measurement(tune.Codec.H264, qs, rates, True)
        self.assertEqual(seen, qs + extra)
        self.assertEqual((gated, scored), (8, 8))
        self.assertEqual(tune.noise_range_status(curve, clip, tune.Codec.H264), "max_qp_floor")
        anchor = tune.Curve([100, 200, 400, 800], [35, 37, 36, 34], [60, 70, 65, 55])
        with self.assertRaisesRegex(ValueError, "noise.*cover 200"):
            tune.noise_deltas({"noise": curve}, {"noise": anchor}, [clip], "psnr")

    def test_noise_extension_is_explicit_and_never_scores_overshooting_points(self):
        qs = [20, 30, 40, 44]
        rates = dict(zip(qs + [46, 47], [600, 400, 300, 220, 1100, 190]))
        short, clip, seen, _, _ = self._mock_noise_measurement(tune.Codec.H264, qs, rates, False)
        self.assertEqual(seen, qs)
        self.assertEqual(tune.noise_range_status(short, clip, tune.Codec.H264), "insufficient_coarse_samples")
        curve, _, seen, gated, scored = self._mock_noise_measurement(tune.Codec.H264, qs, rates, True)
        self.assertEqual(seen, qs + [46, 47])
        self.assertEqual(gated, 6)  # Even rejected rate probes must decode/reconstruct.
        self.assertEqual(scored, 5)
        self.assertNotIn(46, curve.quantizers)
        self.assertTrue(all(rate <= clip.max_test_kbps for rate in curve.kbps))

    def test_noise_metrics_receive_clean_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean, noisy = root / "clean.yuv", root / "noisy.yuv"
            clip = tune.ClipRef("noise", noisy, 4, 4, 30, 1, "synthetic_noise", 10000,
                                clean_reference=clean)
            passed = tune.PointResult(GateResult(True), "recon", None)
            with patch.object(tune, "load_paths"), \
                 patch.object(tune, "run_point", return_value=passed) as run_point:
                tune.encode_point(Path("/binary"), tune.Codec.AV1, clip, 1, 1, {},
                                  root / "s.ivf", metrics=lambda spec: [spec])
            command, _, _, spec, _, metrics, _ = run_point.call_args.args
            self.assertEqual(spec.reference, clean)
            self.assertEqual(metrics, [spec])
            self.assertIn(str(noisy), command("/dev/fd/9"))
            self.assertIn("/dev/fd/9", command("/dev/fd/9"))

    def test_noise_proxy_only_adds_train_noise_with_clean_references(self):
        natural = [SimpleNamespace(name=n, content_class="static") for n in tune.PROXY_CLIPS]
        noise = SimpleNamespace(name="noise", content_class="synthetic_noise",
                                clean_reference=Path("/clean"))
        with patch.object(tune, "load_clips", return_value=natural + [noise]) as load:
            self.assertEqual(tune.proxy_clips(), natural)
            self.assertEqual(tune.proxy_clips(True), natural + [noise])
            load.assert_called_with("train")
            noise.clean_reference = None
            with self.assertRaisesRegex(ValueError, "clean reference"):
                tune.proxy_clips(True)

    def test_key_cost_is_separated_and_amortised(self):
        clips = [SimpleNamespace(name=name) for name in ('a', 'b')]
        # Two clips: I/P costs 200/100M and 400/200M; geometric means.
        with patch.object(tune, 'COST_CLIPS', ('a', 'b')), \
             patch.object(tune, 'load_clips', return_value=clips), \
             patch.object(tune, 'instructions', side_effect=[
                 [200_000_000] + [100_000_000] * tune.COST_P_FRAMES,
                 [400_000_000] + [200_000_000] * tune.COST_P_FRAMES]):
            cost = tune.instruction_cost(Path('/binary'), tune.Codec.H264, {}, Path('/work'))
        root2 = 2 ** 0.5
        self.assertAlmostEqual(cost.p, 100 * root2)
        self.assertAlmostEqual(cost.key, 200 * root2)
        self.assertAlmostEqual(cost.per_frame(1), 200 * root2)
        self.assertAlmostEqual(cost.per_frame(60), (200 + 59 * 100) * root2 / 60)

    def test_cost_weighs_every_clip_equally_in_percent(self):
        clips = [SimpleNamespace(name=name) for name in ('cheap', 'costly')]
        runs = [[1] + [p] * tune.COST_P_FRAMES for p in (50, 200, 55, 200)]
        with patch.object(tune, 'COST_CLIPS', ('cheap', 'costly')), \
             patch.object(tune, 'load_clips', return_value=clips), \
             patch.object(tune, 'instructions', side_effect=runs):
            before = tune.instruction_cost(Path('/binary'), tune.Codec.H264, {}, Path('/work'))
            after = tune.instruction_cost(Path('/binary'), tune.Codec.H264, {}, Path('/work'))
        # +10% on the cheap clip alone is +4.9% overall (sqrt 1.1), not +2%.
        self.assertAlmostEqual(after.p / before.p, 1.1 ** 0.5)

    def test_missing_quality_overlap_rejects_whole_trial(self):
        anchor = tune.Curve([100, 200, 400, 800], [30, 33, 36, 40], [60, 65, 70, 80])
        disjoint = tune.Curve([100, 200, 400, 800], [41, 42, 43, 44], [81, 82, 83, 84])
        for metric in ('psnr', 'vmaf_neg'):
            with self.assertRaisesRegex(ValueError, 'bad_clip.*overlap'):
                tune.bd_by_clip({'good_clip': anchor, 'bad_clip': disjoint},
                                {'good_clip': anchor, 'bad_clip': anchor}, metric)

    def test_rebuilding_same_binary_path_requires_new_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'nanocodec'
            binary.write_bytes(b'build one')
            args = argparse.Namespace(codec='h264', binary=str(binary), params='test.value',
                                      study='guard-test', seed=1, trials=0, workers=1)
            study = MagicMock()
            study.trials = []
            base = tune.Evaluation({}, 1.0)
            with patch.object(tune, 'TUNING_DIR', root), \
                 patch.object(tune, 'specs_of', return_value={'test.value': (1, 0, 2)}), \
                 patch.object(tune, 'proxy_clips', return_value=[]), \
                 patch.object(tune, 'evaluate', return_value=base) as evaluate, \
                 patch.object(tune.optuna, 'create_study', return_value=study), \
                 patch.object(tune, 'print_front'):
                tune.search(args)
                tune.search(args)  # same build: cached anchor is reused
                self.assertEqual(evaluate.call_count, 1)
                binary.write_bytes(b'build two')
                with self.assertRaisesRegex(SystemExit, 'new --study'):
                    tune.search(args)
                self.assertEqual(evaluate.call_count, 1)

    def test_search_space_or_noise_policy_change_requires_new_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "nanocodec"
            binary.write_bytes(b"frozen encoder")
            path = root / "space.json"
            path.write_text(json.dumps({"x": {"choices": [0, 1]}}))
            args = argparse.Namespace(codec="av1", binary=str(binary), params="x",
                                      study="domain-test", seed=1, trials=0, workers=1,
                                      noise_guard=False, noise_report=False, search_space=str(path))
            study = MagicMock()
            study.trials = []
            with patch.object(tune, "TUNING_DIR", root), \
                 patch.object(tune, "specs_of", return_value={"x": (0, 0, 2)}), \
                 patch.object(tune, "proxy_clips", return_value=[]), \
                 patch.object(tune, "evaluate", return_value=tune.Evaluation({}, 1, 1)) as evaluate, \
                 patch.object(tune.optuna, "create_study", return_value=study), \
                 patch.object(tune, "print_front"):
                tune.search(args)
                path.write_text(json.dumps({"x": {"choices": [0, 2]}}))
                with self.assertRaisesRegex(SystemExit, "new --study"):
                    tune.search(args)
                path.write_text(json.dumps({"x": {"choices": [0, 1]}}))
                args.noise_report = True
                with self.assertRaisesRegex(SystemExit, "new --study"):
                    tune.search(args)
                args.noise_report = False
                args.cost_gop = 300
                with self.assertRaisesRegex(SystemExit, "new --study"):
                    tune.search(args)
                args.cost_gop = 0
                args.extend_noise_range = True
                with self.assertRaisesRegex(SystemExit, "new --study"):
                    tune.search(args)
                args.extend_noise_range = False
                anchor_path = root / "domain-test.anchor.json"
                old_metric = json.loads(anchor_path.read_text())
                old_metric["identity"]["vmaf_metric"] = "vmaf_neg"
                anchor_path.write_text(json.dumps(old_metric))
                with self.assertRaisesRegex(SystemExit, "new --study"):
                    tune.search(args)
                self.assertEqual(evaluate.call_count, 1)

    def test_unreliable_overlap_rejects_trial(self):
        a = tune.Curve([100, 200, 400, 800], [30, 33, 36, 40], [60, 65, 70, 80])
        t = tune.Curve([100, 200, 400, 800], [39.9, 43, 46, 50], [79.9, 85, 90, 95])
        for metric in ('psnr', 'vmaf_neg'):
            with self.assertRaisesRegex(ValueError, 'bad_clip.*overlap'):
                tune.bd_by_clip({'bad_clip': t}, {'bad_clip': a}, metric)

    def test_retired_study_cannot_write_presets(self):
        study = SimpleNamespace(user_attrs={'retired_reason': 'outside rate ceiling'})
        with patch.object(tune.optuna, 'load_study', return_value=study):
            with self.assertRaisesRegex(SystemExit, 'retired study.*rate ceiling'):
                tune.write_presets(argparse.Namespace(study='retired'))

    def test_legacy_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'nanocodec'
            binary.write_bytes(b'build')
            (root / 'legacy.anchor.json').write_text(json.dumps({'binary': str(binary)}))
            args = argparse.Namespace(codec='h264', binary=str(binary), params='test.value',
                                      study='legacy')
            with patch.object(tune, 'TUNING_DIR', root), \
                 patch.object(tune, 'specs_of', return_value={'test.value': (1, 0, 2)}), \
                 patch.object(tune, 'proxy_clips', return_value=[]):
                with self.assertRaisesRegex(SystemExit, 'new --study'):
                    tune.search(args)


if __name__ == '__main__':
    unittest.main()


class SeedTrialsTest(unittest.TestCase):
    """--seed-trials points: defaults fill unlisted names; bad seeds are refused."""

    specs = {"a.x": (1.0, 0.0, 2.0), "a.y": (0.5, 0.0, 1.0)}
    domains = {"a.x": {"low": 0.0, "high": 2.0}, "a.y": {"choices": [0.0, 0.5, 1.0]}}

    def seeds(self, content):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(content, f)
        return tune.seed_trials(f.name, ["a.x", "a.y"], self.specs, self.domains)

    def test_unlisted_parameters_take_defaults(self):
        self.assertEqual(self.seeds([{"a.x": 1.5}]), [{"a.x": 1.5, "a.y": 0.5}])

    def test_refuses_unknown_names_and_values_outside_the_space(self):
        with self.assertRaises(SystemExit):
            self.seeds([{"a.z": 1.0}])
        with self.assertRaises(SystemExit):
            self.seeds([{"a.y": 0.7}])

    def test_no_file_means_no_seeds(self):
        self.assertEqual(tune.seed_trials(None, ["a.x"], self.specs, self.domains), [])
