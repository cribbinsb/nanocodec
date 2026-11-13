"""Screen profiles must never masquerade as VMAF or lose completed curves."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nceval import tune
from fake_point import FakeEncodePoint

class ScreeningTests(unittest.TestCase):
    def test_screen_search_reuses_default_anchor_and_names_metrics_honestly(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);binary=w/'encoder';binary.write_bytes(b'encoder')
            clips=[tune.ClipRef(n,w/n,256,256,30,150,'static',2000) for n in tune.COST_CLIPS]
            curve=tune.Curve([10,20,30,40],[30,31,32,33],quantizers=[1,2,3,4],ms_ssim=[.9,.92,.94,.96])
            study=tune.optuna.create_study(directions=['minimize']*3,storage=f'sqlite:///{w}/study.sqlite')
            study.enqueue_trial({'test.value':1})
            args=argparse.Namespace(codec='hevc',binary=str(binary),params='test.value',study='screen',seed=1,trials=1,workers=1,metric_profile='psnr-ms-ssim',noise_report=False,noise_guard=False,cost_p_frames=48)
            with patch.object(tune,'TUNING_DIR',w),patch.object(tune,'specs_of',return_value={'test.value':(1,0,2)}),patch.object(tune,'proxy_clips',return_value=clips),patch.object(tune,'clip_fingerprint',return_value='input'),patch.object(tune,'evaluate',return_value=tune.Evaluation({c.name:curve for c in clips},1,1)) as evaluate,patch.object(tune.optuna,'create_study',return_value=study),patch.object(tune,'print_front'):
                tune.search(args)
            self.assertEqual(evaluate.call_count,1)
            trial=study.trials[0];self.assertEqual(trial.values,[0,0,1]);self.assertIn('anchor_reused',trial.user_attrs)
            self.assertIn('class_bd_ms_ssim',trial.user_attrs);self.assertNotIn('class_bd_vmaf_v1',trial.user_attrs)
            anchor=json.loads((w/'screen.anchor.json').read_text());self.assertEqual(anchor['identity']['metric_profile'],'psnr-ms-ssim');self.assertEqual(anchor['identity']['cost_p_frames'],48)
            self.assertIn('ms_ssim',anchor['curves'][clips[0].name]);self.assertNotIn('vmaf_v1',anchor['curves'][clips[0].name])

    def test_screen_record_has_distinct_raw_ms_ssim_field(self):
        c=tune.Curve([10,20,30,40],[30,31,32,33],quantizers=[4,3,2,1],ms_ssim=[.9,.92,.94,.96])
        record=tune.curve_record(c,tune.SCREEN_METRIC_PROFILE)
        self.assertIn('ms_ssim',record)
        self.assertNotIn('vmaf_v1',record)
        with self.assertRaises(ValueError):tune.curve_record(c)

    def test_screen_keeps_only_compressed_certificates_and_never_calls_vmaf(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);binary=w/'encoder';binary.write_bytes(b'encoder')
            c=tune.ClipRef('one',w/'source.yuv',256,256,30,2,'static',2000)
            encode=FakeEncodePoint(stream_bytes=lambda q:b'compressed')
            with patch.object(tune,'encode_point',side_effect=encode),patch.object(tune,'file_sha256',return_value='binary'),patch.object(tune,'payload_bytes',return_value=100),patch.object(tune,'clip_fingerprint',return_value='source'),patch.object(tune,'check_disk'):
                curve=tune.measure_curve(binary,tune.Codec.HEVC,c,2,{},w,[4,3,2,1],metric_profile=tune.SCREEN_METRIC_PROFILE,keep_streams=True)
            spec=tune.StreamSpec(c.yuv,c.fmt,2,30,SimpleNamespace(anchors=w))
            for call in encode.calls:
                kinds={type(m).__name__ for m in call['metrics'](spec)}
                self.assertEqual(kinds,{'PsnrMetric','MsSsimMetric'})
            self.assertEqual(curve.quantizers,[1,2,3,4]);self.assertEqual(curve.ms_ssim,[.95]*4);self.assertEqual(curve.vmaf_v1,[])
            manifests=list((w/'retained').glob('*.json'));self.assertEqual(len(manifests),4)
            for p in manifests:
                m=json.loads(p.read_text());self.assertEqual(m['metric_profile'],'psnr-ms-ssim');self.assertEqual(m['ms_ssim'],.95);self.assertIn('recon_sha256',m)
            self.assertFalse(list(w.glob('*.yuv')))

    def test_curve_checkpoint_survives_cost_failure_and_is_reused(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);binary=w/'encoder';binary.write_bytes(b'encoder');work=w/'work';c=tune.ClipRef('one',w/'source',256,256,30,150,'static',2000)
            curve=tune.Curve([10,20,30,40],[30,31,32,33],quantizers=[4,3,2,1],ms_ssim=[.9,.92,.94,.96])
            kwargs=dict(clips=[c],metric_profile=tune.SCREEN_METRIC_PROFILE)
            with patch.object(tune,'COST_CLIPS',('one',)),patch.object(tune,'clip_fingerprint',return_value='source'),patch.object(tune,'measure_curve',return_value=curve) as measure,patch.object(tune,'instruction_cost',side_effect=RuntimeError('cost interrupted')):
                with self.assertRaises(RuntimeError):tune.evaluate(binary,tune.Codec.HEVC,{},work,**kwargs)
                self.assertEqual(measure.call_count,1)
            self.assertTrue((work/'curve-checkpoints/one.json').exists())
            with patch.object(tune,'COST_CLIPS',('one',)),patch.object(tune,'clip_fingerprint',return_value='source'),patch.object(tune,'measure_curve') as measure,patch.object(tune,'instruction_cost',return_value=tune.InstructionCost(1,1)):
                result=tune.evaluate(binary,tune.Codec.HEVC,{},work,**kwargs)
                measure.assert_not_called();self.assertEqual(result.curves['one'].ms_ssim,curve.ms_ssim)

    def test_subset_rejects_val_and_missing_classes_and_adds_all_noise(self):
        names=[*tune.COST_CLIPS,'low','pan'];classes=['crowd','bodycam','static','lowlight','pan']
        clips=[tune.ClipRef(n,Path('/'+n),256,256,30,150,k,2000) for n,k in zip(names,classes)]
        clips+=[tune.ClipRef('noise'+str(i),Path('/noise'),256,256,30,150,'synthetic_noise',2000,clean_reference=Path('/clean')) for i in range(6)]
        with patch.object(tune,'load_clips',return_value=clips):
            self.assertEqual(len(tune.selected_train_clips(','.join(names),True)),11)
            with self.assertRaises(ValueError):tune.selected_train_clips(','.join(names[:-1]),False)
            with self.assertRaises(ValueError):tune.selected_train_clips(','.join(names)+',val',False)

    def test_checkpoint_rejects_clean_reference_and_protocol_changes(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);binary=w/'encoder';binary.write_bytes(b'encoder')
            c=tune.ClipRef('noise',w/'noise',256,256,30,150,'synthetic_noise',2000,clean_reference=w/'clean')
            curve=tune.Curve([10,20,30,40],[30,31,32,33],quantizers=[1,2,3,4],ms_ssim=[.9,.92,.94,.96])
            kwargs=dict(metric_profile=tune.SCREEN_METRIC_PROFILE,identity={'protocol':'a'})
            with patch.object(tune,'clip_fingerprint',side_effect=lambda c:str(c.yuv)),patch.object(tune,'measure_curve',return_value=curve):
                tune.checkpointed_curve(binary,tune.Codec.HEVC,c,{},w,**kwargs)
                changed=tune.replace(c,clean_reference=w/'different-clean')
                with self.assertRaises(ValueError):tune.checkpointed_curve(binary,tune.Codec.HEVC,changed,{},w,**kwargs)
                kwargs['identity']={'protocol':'b'}
                with self.assertRaises(ValueError):tune.checkpointed_curve(binary,tune.Codec.HEVC,c,{},w,**kwargs)

    def test_explicit_cost_duration_has_rate_preflight_before_callgrind(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);c=tune.ClipRef('one',w/'source',256,256,30,150,'static',2000)
            encode=FakeEncodePoint()
            with patch.object(tune,'encode_point',side_effect=encode),patch.object(tune,'file_sha256',return_value='binary'),patch.object(tune,'run') as callgrind,patch.object(tune,'probe_frames',return_value=[SimpleNamespace(bytes=10**6)]*49):
                with self.assertRaisesRegex(ValueError,'source cap'):
                    tune.instructions(w/'encoder',tune.Codec.HEVC,c,49,{},w,40,enforce_rate_cap=True)
                self.assertEqual(len(encode.calls),1);callgrind.assert_not_called()
            with patch.object(tune,'load_clips',return_value=[c]),patch.object(tune,'COST_CLIPS',('one',)),patch.object(tune,'instructions',side_effect=[[100]+[100]*48]) as calls:
                cost=tune.instruction_cost(w/'encoder',tune.Codec.HEVC,{},w,{'one':40},p_frames=48)
                self.assertEqual([x.args[3] for x in calls.call_args_list],[49])
                self.assertTrue(all(x.kwargs['enforce_rate_cap'] for x in calls.call_args_list))
                self.assertAlmostEqual(cost.p,.0001)

    def test_cost_probe_cap_ignores_the_amortized_key_frame(self):
        with tempfile.TemporaryDirectory() as t:
            w=Path(t);c=tune.ClipRef('one',w/'source',256,256,30,150,'static',2000)
            sizes=[SimpleNamespace(bytes=10**6)]+[SimpleNamespace(bytes=1000)]*6
            with patch.object(tune,'encode_point',side_effect=FakeEncodePoint()),patch.object(tune,'probe_frames',return_value=sizes),\
                 patch.object(tune,'run'),patch.object(tune,'_callgrind_total',return_value=5):
                (w/'cost_one.hevc').write_bytes(b'x')
                self.assertEqual(tune.instructions(w/'encoder',tune.Codec.HEVC,c,7,{},w,40,enforce_rate_cap=True),5)

    def test_screen_front_uses_ms_ssim_label_and_cannot_export(self):
        trial=SimpleNamespace(number=0,values=[-1,-2,.9],params={},state=tune.optuna.trial.TrialState.COMPLETE,user_attrs={'constraint':[0,0]})
        study=SimpleNamespace(study_name='screen',trials=[trial],user_attrs={'metric_profile':'psnr-ms-ssim'})
        with patch.object(tune,'vmaf_label') as label:
            tune.print_front(study);label.assert_not_called()
        with patch.object(tune.optuna,'load_study',return_value=study),patch.object(tune,'study_storage'):
            with self.assertRaises(SystemExit):tune.write_presets(SimpleNamespace(study='screen'))

if __name__=='__main__':unittest.main()
