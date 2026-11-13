"""V1 preprocessing/model identity and legacy storage/report isolation."""
import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import gpuserve, metrics, fingerprint, report, tune
from nceval.clips import Clip, ClipSet
from nceval.config import Paths
from nceval.framestats import FrameInfo
from nceval.results import ResultsDb, EncodeRecord, SCHEMA
from nceval.sweep import SweepConfig
from nceval.yuv import YuvFormat


def model_paths(tmp_path):
    paths = Paths(tmp_path / 'data', tmp_path / 'anchors', tmp_path / 'bin')
    model = paths.anchors / metrics.VMAF_V1_MODEL
    model.parent.mkdir(parents=True)
    model.write_text('{"test_model":1}')
    return paths, model


def test_v1_geometry_preserves_orientation_aspect_and_even_420_dimensions():
    for w, h, expected in [(1280, 720, (1920, 1080)), (720, 1280, (1080, 1920)),
                           (854, 480, (1922, 1080)), (320, 320, (1080, 1080))]:
        width, height = metrics.vmaf_v1_geometry(YuvFormat(w, h))
        assert (width, height) == expected
        assert width % 2 == height % 2 == 0


class RecordingProcesses:
    """A ProcessSet stand-in that records spawn() calls instead of running them."""

    def __init__(self):
        self.calls = []

    def spawn(self, cmd, label, **kwargs):
        self.calls.append((cmd, label, kwargs))
        return SimpleNamespace(stdin=io.BytesIO())


def test_native_v1_feeds_libvmaf_directly_at_encoded_geometry(tmp_path):
    paths, model = model_paths(tmp_path)
    reference = tmp_path / 'clip.yuv'
    spec = metrics.StreamSpec(reference, YuvFormat(1280, 720), 150, 30, paths)
    processes = RecordingProcesses()
    with patch.object(metrics, 'find_tool', side_effect=lambda name, _: Path('/tools') / name):
        sinks = metrics.VmafV1Metric(spec).start(processes, tmp_path)
    assert len(sinks) == 1
    ((vmaf, label, kwargs),) = processes.calls  # libvmaf only: no ffmpeg
    assert Path(vmaf[0]).name == 'vmaf' and label == 'vmaf_v1'
    assert vmaf[vmaf.index('-r') + 1] == str(reference) and vmaf[vmaf.index('-d') + 1] == '/dev/stdin'
    assert [vmaf[vmaf.index(o) + 1] for o in ('-w', '-h', '-b', '--frame_cnt', '--subsample')] == \
        ['1280', '720', '8', '150', '1']
    assert f'path={model}:name=vmaf_v1:cambi.enc_width=1280:cambi.enc_height=720:cambi.enc_bitdepth=8' \
        == vmaf[vmaf.index('--model') + 1]


def test_v1_display_streams_10bit_upscales_with_original_encode_cambi_hints(tmp_path):
    paths, _ = model_paths(tmp_path)
    reference = tmp_path / 'clip.yuv'
    spec = metrics.StreamSpec(reference, YuvFormat(1280, 720), 150, 30, paths)
    processes = RecordingProcesses()
    with patch.object(metrics, 'find_tool', side_effect=lambda name, _: Path('/tools') / name):
        sinks = metrics.VmafV1DisplayMetric(spec).start(processes, tmp_path)
    assert len(sinks) == 1
    (ref_up, _, ref_kw), (dist_up, _, dist_kw), (vmaf, _, vmaf_kw) = processes.calls
    # The reference upscale reads only the encoded prefix of the clip itself.
    assert ref_up[ref_up.index('-i') + 1] == str(reference)
    assert ref_up[ref_up.index('-frames:v') + 1] == '150'
    assert dist_up[dist_up.index('-i') + 1] == 'pipe:0' and '-frames:v' not in dist_up
    for command in (ref_up, dist_up):
        assert 'scale=1920:1080:flags=bicubic,format=yuv420p10le' in command
        assert command[-1] == 'pipe:1'  # never a display copy on disk
    assert vmaf[vmaf.index('-b') + 1] == '10'
    assert vmaf[vmaf.index('--subsample') + 1] == '1'
    assert 'name=vmaf_v1:cambi.enc_width=1280:cambi.enc_height=720:cambi.enc_bitdepth=8' in vmaf[vmaf.index('--model') + 1]
    fds = vmaf_kw['pass_fds']
    assert [vmaf[vmaf.index('-r') + 1], vmaf[vmaf.index('-d') + 1]] == [f'/dev/fd/{fd}' for fd in fds]


def test_v1_report_is_sorted_by_frame(tmp_path):
    report = tmp_path / 'vmaf.json'
    report.write_text(json.dumps({
        'pooled_metrics': {'vmaf_v1': {'mean': 90.5}},
        'frames': [{'frameNum': 1, 'metrics': {'vmaf_v1': 91}},
                   {'frameNum': 0, 'metrics': {'vmaf_v1': 90}}]}))
    result = metrics._parse_vmaf_v1(report)
    assert result.summary == {'vmaf_v1': 90.5}
    assert result.per_frame == [{'vmaf_v1': 90.0}, {'vmaf_v1': 91.0}]


def test_model_bytes_and_preprocessing_revision_change_v1_not_legacy_identity(tmp_path):
    paths, model = model_paths(tmp_path)
    with patch.object(metrics, 'find_tool', return_value=model), \
         patch.object(metrics.subprocess, 'run', return_value=SimpleNamespace(stdout='ffmpeg')):
        fp = lambda profile: fingerprint.measurement_fingerprint(profile, paths)
        native, display, legacy = fp(metrics.V1_PROFILE), fp(metrics.DISPLAY_PROFILE), fp(metrics.LEGACY_PROFILE)
        assert len({native, display, legacy}) == 3  # profiles never share results
        model.write_text('{"test_model":2}')
        assert fp(metrics.V1_PROFILE) != native and fp(metrics.DISPLAY_PROFILE) != display
        native, display = fp(metrics.V1_PROFILE), fp(metrics.DISPLAY_PROFILE)
        with patch.object(metrics, 'VMAF_V1_PREPROCESS_REVISION', 2):
            assert fp(metrics.DISPLAY_PROFILE) != display
            assert fp(metrics.V1_PROFILE) == native  # native has no preprocessing
        assert fp(metrics.LEGACY_PROFILE) == legacy


def record(structure):
    return EncodeRecord('enc', structure, 'clip', 30, 1, 10, 100, True, '', 'test', 'fp', 'test')


def test_old_database_keeps_legacy_values_and_migrates_per_frame_v1_fields(tmp_path):
    path = tmp_path / 'results.sqlite'
    old_schema = SCHEMA.replace(', vmaf_v1 REAL, ms_ssim REAL', '')
    old_schema = old_schema.replace("    gate_level TEXT NOT NULL DEFAULT '',\n", '')
    assert 'gate_level' not in old_schema
    with sqlite3.connect(path) as conn:
        conn.executescript(old_schema)
    db = ResultsDb(path)
    db.store(record('ip10s-tl1'), {'vmaf_neg': 77}, [FrameInfo(10, 'I')], [{'vmaf_neg': 77}])
    db.store(record('ip10s-tl1-vmaf-v1'), {'vmaf_v1': 91, 'ms_ssim': .95},
             [FrameInfo(10, 'I')], [{'vmaf_v1': 91, 'ms_ssim': .95}])
    assert db.rd_points('enc', 'ip10s-tl1', 'clip', 'test', 'vmaf_neg') == [(100, 77)]
    assert db.rd_points('enc', 'ip10s-tl1-vmaf-v1', 'clip', 'test', 'vmaf_v1') == [(100, 91)]
    assert db.conn.execute('select vmaf_neg,vmaf_v1,ms_ssim from frames order by encode_id').fetchall() == [(77, None, None), (None, 91, .95)]
    db.conn.close()


def test_namespace_and_legacy_metric_report_lookup_do_not_mix(tmp_path):
    assert SweepConfig(10).structure_tag == 'ip10s-tl1-vmaf-v1n'
    assert SweepConfig(10, measurement_profile='vmaf-v1-display').structure_tag == 'ip10s-tl1-vmaf-v1'
    assert SweepConfig(10, measurement_profile='legacy').structure_tag == 'ip10s-tl1'
    db = ResultsDb(tmp_path / 'results.sqlite')
    for encoder in ['anchor', 'candidate']:
        for q, rate, quality in [(10, 100, 30), (20, 200, 33), (30, 400, 36), (40, 800, 40)]:
            for structure, metric, score in [('ip10s-tl1', 'vmaf_neg', quality),
                                              ('ip10s-tl1-vmaf-v1', 'vmaf_v1', quality + 10)]:
                rec = EncodeRecord(encoder, structure, 'clip', q, 1, 10, rate, True, '', 'test', 'fp', 'test')
                db.store(rec, {metric: score})
    clips = ClipSet('test', (Clip('clip', Path('/clip'), 1280, 720, 30, 1, 2000, 'static', 'val'),))
    for metric in ['vmaf_neg', 'vmaf_v1']:
        rows = report.compute(db, clips, 'anchor', metric, 'ip10s-tl1-vmaf-v1', 'test', ['candidate'])
        assert abs(rows[0].result.bd_rate_percent) < 1e-8
    db.conn.close()


def test_new_curve_storage_never_renames_legacy_values():
    old = tune.Curve([100], [30], [77], [20])
    try:
        tune.curve_record(old)
    except ValueError as error:
        assert 'legacy' in str(error)
    else:
        assert False, 'legacy curve must not masquerade as v1'
    new = tune.Curve([100], [30], quantizers=[20], vmaf_v1=[91])
    assert tune.curve_record(new) == {'kbps': [100], 'psnr': [30], 'quantizers': [20], 'vmaf_v1': [91]}


def test_ms_ssim_is_explicit_not_default_sweep_work(tmp_path):
    paths, _ = model_paths(tmp_path)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(320, 240), 2, 30, paths)
    kinds = lambda components: [type(c).__name__ for c in components]
    assert kinds(metrics.profile_metrics(spec)) == ['PsnrMetric', 'VmafV1Metric']  # XPSNR opt-in
    assert kinds(metrics.profile_metrics(spec, include_xpsnr=True)) == ['PsnrMetric', 'XpsnrMetric', 'VmafV1Metric']
    assert kinds(metrics.profile_metrics(spec, metrics.DISPLAY_PROFILE)) == ['PsnrMetric', 'XpsnrMetric', 'VmafV1DisplayMetric']
    explicit = metrics.profile_metrics(spec, include_ms_ssim=True)
    assert kinds(explicit)[-1] == 'MsSsimMetric'
    legacy = metrics.profile_metrics(spec, metrics.LEGACY_PROFILE)
    assert legacy[-1]._label == 'vmaf_legacy'


def test_tuner_calls_shared_v1_protocol_and_legacy_labels_remain_explicit(tmp_path):
    from types import SimpleNamespace
    paths, _ = model_paths(tmp_path)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(1280, 720), 2, 30, paths)
    psnr, vmaf = tune.point_metrics(tune.V1_METRIC_PROFILE)(spec)
    assert isinstance(psnr, metrics.PsnrMetric) and isinstance(vmaf, metrics.VmafV1Metric)
    assert vmaf._subsample == tune.VMAF_SUBSAMPLE == 1
    assert tune.study_vmaf_metric(SimpleNamespace(user_attrs={})) == 'vmaf_neg'
    assert 'legacy' in tune.vmaf_label('vmaf_neg')
    assert 'v1.0.16' in tune.vmaf_label('vmaf_v1')


def test_cli_defaults_new_namespace_and_autodetects_explicit_legacy_metric(tmp_path):
    from nceval import __main__ as cli
    paths, _ = model_paths(tmp_path)
    for extra, expected in [([], 'ip10s-tl1-vmaf-v1n'),
                            (['--vmaf-display'], 'ip10s-tl1-vmaf-v1'),
                            (['--metric', 'vmaf_neg'], 'ip10s-tl1'),
                            (['--legacy-metrics', '--metric', 'psnr_yuv'], 'ip10s-tl1')]:
        args = cli._parser().parse_args(['report', '--clipset', '/clips'] + extra)
        with patch.object(cli, 'load_paths', return_value=paths), \
             patch.object(cli, 'load_clipset', return_value=ClipSet('empty', ())):
            _, db, _, cfg = cli._setup(args)
        assert cfg.structure_tag == expected
        db.conn.close()


def test_preprocessing_ffmpeg_binary_and_library_build_change_protocol(tmp_path):
    paths, _ = model_paths(tmp_path)
    tool = tmp_path / "ffmpeg"
    tool.write_bytes(b"binary-one")
    with patch.object(metrics, "find_tool", return_value=tool), \
         patch.object(metrics.subprocess, "run", return_value=SimpleNamespace(stdout="build and libs one")):
        first = metrics.vmaf_v1_protocol(paths, display=True)
        tool.write_bytes(b"binary-two")
        assert metrics.vmaf_v1_protocol(paths, display=True) != first
        second = metrics.vmaf_v1_protocol(paths, display=True)
    with patch.object(metrics, "find_tool", return_value=tool), \
         patch.object(metrics.subprocess, "run", return_value=SimpleNamespace(stdout="build and libs two")):
        assert metrics.vmaf_v1_protocol(paths, display=True) != second


def test_real_metric_set_is_psnr_one_vmaf_pass_and_every_frame_ms_ssim(tmp_path):
    paths, model = model_paths(tmp_path)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(640, 360), 150, 30, paths)
    psnr, vmaf, ms_ssim = metrics.real_metrics(spec)
    assert isinstance(psnr, metrics.PsnrMetric) and isinstance(ms_ssim, metrics.MsSsimMetric)
    processes = RecordingProcesses()
    with patch.object(metrics, 'find_tool', side_effect=lambda name, _: Path('/tools') / name):
        vmaf.start(processes, tmp_path)
    (pass_one, _, _), = processes.calls  # MS-SSIM runs in-process, not in libvmaf
    models = [pass_one[i + 1] for i, a in enumerate(pass_one) if a == '--model']
    assert len(models) == 1 and models[0].startswith(f'path={model}:name=vmaf_v1:')
    ms_ssim._sink = SimpleNamespace(scores=[0.96, 0.98])
    measured = ms_ssim.measurement(processes)
    assert measured.summary == {'ms_ssim': pytest.approx(0.97)}
    assert measured.per_frame == [{'ms_ssim': 0.96}, {'ms_ssim': 0.98}]


def fake_vmafx(tmp_path, monkeypatch):
    """Select the VMAFx backend with a dummy tool and no GPU self-check."""
    vmafx = tmp_path / 'vmafx'
    vmafx.write_text('#!/bin/sh\n')
    vmafx.chmod(0o755)
    monkeypatch.setenv(metrics.METRICS_BACKEND_ENV, metrics.VMAFX_CUDA_BACKEND)
    monkeypatch.setenv(metrics.VMAFX_ENV, str(vmafx))
    monkeypatch.setattr(metrics, '_vmafx_self_check', lambda tool, model: None)
    return vmafx


def vmafx_report(frames, backends=('adm_cuda', 'float_ms_ssim_cuda'), score=93.1234567):
    return {'feature_backends': [{'extractor': name, 'backend': name.rsplit('_', 1)[-1]
                                  if name.endswith('_cpu') else 'cuda'} for name in backends],
            'pooled_metrics': {'vmaf_v1': {'mean': score}},
            'frames': [{'frameNum': i, 'metrics': {'vmaf_v1': score, 'float_ms_ssim': 0.9 + i / 100}}
                       for i in range(frames)]}


def test_vmafx_backend_scores_v1_and_ms_ssim_in_one_gpu_process(tmp_path, monkeypatch):
    paths, model = model_paths(tmp_path)
    vmafx = fake_vmafx(tmp_path, monkeypatch)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(640, 360), 6, 30, paths)
    psnr, gpu = metrics.real_metrics(spec)
    assert isinstance(gpu, metrics.VmafxMetric)
    assert gpu.cpu_threads == metrics.GPU_METRIC_SLOT_WEIGHT and gpu.uses_gpu
    assert metrics.gpu_slot_need([psnr, gpu]) == 1 and metrics.gpu_slot_need([psnr]) == 0
    processes = RecordingProcesses()
    gpu.start(processes, tmp_path)
    (cmd, _, _), = processes.calls
    assert cmd[0] == str(vmafx)
    assert cmd[cmd.index('--backend') + 1] == 'cuda' and cmd[cmd.index('--precision') + 1] == 'max'
    assert cmd[cmd.index('--model') + 1].startswith(f'path={model}:name=vmaf_v1:')
    assert cmd[cmd.index('--feature') + 1] == 'float_ms_ssim' and '--threads' not in cmd
    measured = gpu._parse_report(vmafx_report(6))
    # VMAF at libvmaf's %.6f; MS-SSIM on every frame.
    assert measured.summary == {'vmaf_v1': 93.123457, 'ms_ssim': pytest.approx(0.925)}
    assert measured.per_frame[1] == {'vmaf_v1': 93.123457, 'ms_ssim': pytest.approx(0.91)}


def test_vmafx_runs_go_to_a_scoring_server_of_this_build(tmp_path, monkeypatch):
    paths, _ = model_paths(tmp_path)
    vmafx = fake_vmafx(tmp_path, monkeypatch)
    monkeypatch.setenv(gpuserve.POOL_SIZE_ENV, '3')
    names = set()
    for _ in range(60):
        command = metrics.vmafx_command(paths)
        assert command[:2] == [str(vmafx), '--via-server']
        names.add(command[2])
    assert names == {gpuserve.server_name(vmafx, i) for i in range(3)}
    # A rebuilt tool never reaches a server of the old build.
    stat = vmafx.stat()
    os.utime(vmafx, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert gpuserve.server_name(vmafx, 0) not in names
    monkeypatch.setenv(metrics.VMAFX_SERVER_ENV, '0')
    assert metrics.vmafx_command(paths) == [str(vmafx)]


def test_a_metric_dying_under_the_encoder_fails_the_point_loudly(tmp_path):
    """A VMAFx server that dies mid-run (OOM, crash) breaks the pipe the
    encoder writes its reconstruction into. That must surface as
    MetricFailed, not as the encoder failing (which nceval.realuse records
    as a refused, i.e. silently missing, point)."""
    from nceval import point
    from nceval.framepipe import ProcessFailed
    fmt = YuvFormat(64, 64)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', fmt, 50, 30, None)
    dying = tmp_path / 'dying-metric'
    dying.write_text('#!/bin/sh\nhead -c 100 > /dev/null\necho "server ended during the job" >&2\n'
                     'exit 1\n')
    dying.chmod(0o755)

    class DyingMetric:
        cpu_threads = 0
        uses_gpu = False

        def start(self, processes, scratch):
            return [metrics.ProcessSink(processes.spawn([str(dying)], 'vmafx',
                                                        stdin=subprocess.PIPE))]

        def measurement(self, processes):
            raise AssertionError('never reached')

    writer = tmp_path / 'encoder.py'
    writer.write_text('import sys\nout = open(sys.argv[1], "wb", buffering=0)\n'
                      f'for _ in range(50): out.write(bytes({fmt.frame_size}))\n')
    with pytest.raises(metrics.MetricFailed, match='server ended during the job'):
        point.score_ungated(lambda recon: [sys.executable, str(writer), recon], spec,
                            [DyingMetric()], None)
    # An encoder that fails on its own stays an encoder failure.
    failing = tmp_path / 'refusing.py'
    failing.write_text('import sys\nsys.exit("unsupported configuration")\n')

    class QuietMetric(DyingMetric):
        def start(self, processes, scratch):
            return [metrics.ProcessSink(processes.spawn(['cat'], 'vmafx', stdin=subprocess.PIPE,
                                                        stdout=subprocess.DEVNULL))]
    with pytest.raises(ProcessFailed, match='unsupported configuration'):
        point.score_ungated(lambda recon: [sys.executable, str(failing), recon], spec,
                            [QuietMetric()], None)


def test_vmafx_report_must_be_cuda_complete_and_finite(tmp_path, monkeypatch):
    paths, _ = model_paths(tmp_path)
    fake_vmafx(tmp_path, monkeypatch)
    spec = metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(640, 360), 3, 30, paths)
    gpu = metrics.VmafxMetric(spec, ms_ssim=True)
    for bad in (vmafx_report(2),  # a frame short
                vmafx_report(3, backends=('adm_cuda',)),  # MS-SSIM not on CUDA
                vmafx_report(3, backends=('adm_cuda', 'float_ms_ssim_cuda', 'cambi_cpu')),
                vmafx_report(3, score=float('nan'))):
        with pytest.raises(RuntimeError):
            gpu._parse_report(bad)


def test_point_measurement_rejects_short_or_non_finite_rows():
    good = metrics.Measurement({'vmaf_v1': 90.0}, [{'vmaf_v1': 90.0}] * 2)
    metrics.check_measurement(good, 2)
    with pytest.raises(RuntimeError):
        metrics.check_measurement(good, 3)
    with pytest.raises(RuntimeError):
        metrics.check_measurement(metrics.Measurement({'ms_ssim': float('nan')}), None)
    metrics.check_measurement(metrics.Measurement({'xpsnr_y': float('inf')}), None)


def test_metrics_backend_defaults_to_vmafx_and_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv(metrics.METRICS_BACKEND_ENV, raising=False)
    assert metrics.metrics_backend() == metrics.VMAFX_CUDA_BACKEND
    monkeypatch.setenv(metrics.METRICS_BACKEND_ENV, 'opencl')
    with pytest.raises(ValueError):
        metrics.metrics_backend()
    monkeypatch.setenv(metrics.METRICS_BACKEND_ENV, metrics.VMAFX_CUDA_BACKEND)
    monkeypatch.setenv(metrics.VMAFX_ENV, str(tmp_path / 'missing'))
    paths, _ = model_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        metrics.real_metrics(metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(64, 64), 1, 30, paths))
    broken = tmp_path / 'broken'
    broken.write_text('#!/bin/sh\nexit 3\n')
    broken.chmod(0o755)
    monkeypatch.setenv(metrics.VMAFX_ENV, str(broken))
    with pytest.raises(RuntimeError, match='self-check'):
        metrics.real_metrics(metrics.StreamSpec(tmp_path / 'clip.yuv', YuvFormat(64, 64), 1, 30, paths))
