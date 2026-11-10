"""Generate a complete nonzero-offset slider benchmark, estimate, and score."""
from run_case import ROOT,save,sha,freeze
import core
import slider_support as support
import numpy as np
import json,argparse

def main(tag):
    out=ROOT/'runs'/tag;cfg=json.loads((ROOT/'configs'/__import__('os').environ.get('STEP_SLIDER_CONFIG','slider.json')).read_text())
    freeze(out,cfg)
    save(out/'SLIDER_CHOICES.json',dict(mounting='Prescribed nonzero IMU-to-C lever in IMU3 frame.',
        slider_lever_m=cfg['slider_lever'],axes='Stage II A/B enabled; C has no separate axis update.',
        initial_heading='Estimated Body1 axis horizontal projection defines +Z; no true attitude input.',
        attitude_target='Stage I R30, not identity or ground truth.',guide='Prescribed c=origin, estimated d; applies to C.'))
    support.HERE=out;(out/'results').mkdir()
    closure=support.generate_data(support.get_generator(ROOT/'src/generate_slider.py'),cfg)
    with np.load(out/'data/imu_measurements.npz') as z:m={k:z[k] for k in z.files}
    def checkpoint(b,cal,h,timing):
        save(out/'results'/f'body{b}_stageI.json',cal);save(out/'results'/f'body{b}_checks.json',h['checks'])
        np.savez_compressed(out/'results'/f'body{b}_states.npz',time=m['time'],**{k:v for k,v in h.items() if isinstance(v,np.ndarray)})
        assert all(np.array_equal(h[k][0],cal[q]) for k,q in [('R','R0'),('p','p0'),('v','v0')])
    c,h,t=core.run_pipeline(m,cfg,checkpoint)
    save(out/'results/PRE_EVALUATION_HASHES.json',{p.name:sha(p) for p in (out/'results').iterdir() if p.is_file()})
    with np.load(out/'data/evaluation_truth.npz') as ref:rows,geo=support.evaluate(c,h,ref,m['time'])
    save(out/'timing.json',t)
    save(out/'validation.json',dict(analytic_closure=closure,**support.test_jacobians()))
    save(out/'STATUS.json',dict(computation_complete=True,all_optimizers_converged=all(x['fit']['converged'] for x in c.values()),accuracy_certified=False))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();main(a.tag)
