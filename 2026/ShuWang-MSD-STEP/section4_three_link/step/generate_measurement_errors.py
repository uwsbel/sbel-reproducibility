"""Add reproducible sensor errors to the unchanged clean trajectory, once."""
from workflow_io import ROOT,save,sha,load_npz
import numpy as np,json


def generate_arrays(clean,cfg):
    scenarios={name:{k:v.copy() for k,v in clean.items()} for name in ['noise_only','bias_only','noise_bias']}
    draws={}
    for b in [1,2,3]:
        rng=np.random.default_rng(cfg['body_seeds'][str(b)])
        ba=rng.uniform(*cfg['acc_bias_component_bounds_m_s2'],3)
        bg=rng.uniform(*cfg['gyro_bias_component_bounds_rad_s'],3)
        na=rng.normal(0,cfg['acc_white_noise_sd_m_s2'],clean[f'acc{b}'].shape)
        ng=rng.normal(0,cfg['gyro_white_noise_sd_rad_s'],clean[f'gyr{b}'].shape)
        draws.update({f'ba{b}':ba,f'bg{b}':bg,f'noise_acc{b}':na,f'noise_gyr{b}':ng})
        for prefix,bias,noise in [('acc',ba,na),('gyr',bg,ng)]:
            key=f'{prefix}{b}';scenarios['noise_only'][key]=clean[key]+noise
            scenarios['bias_only'][key]=clean[key]+bias
            scenarios['noise_bias'][key]=clean[key]+noise+bias
    return scenarios,draws


def main():
    cfg=json.loads((ROOT/'configs/measurement_errors.json').read_text());clean=load_npz(ROOT/'data/clean/three_link_checked_measurements.npz')
    reference=load_npz(ROOT/'data/clean/three_link_checked_reference.npz');scenarios,draws=generate_arrays(clean,cfg)
    for name in scenarios:
        if (ROOT/'data'/name).exists():raise FileExistsError(ROOT/'data'/name)
    audit=[]
    for name,m in scenarios.items():
        dest=ROOT/'data'/name;dest.mkdir();ref={k:v.copy() for k,v in reference.items()}
        for b in [1,2,3]:
            ref[f'ba{b}']=draws[f'ba{b}'].copy() if name!='noise_only' else np.zeros(3)
            ref[f'bg{b}']=draws[f'bg{b}'].copy() if name!='noise_only' else np.zeros(3)
        np.savez_compressed(dest/'measurements.npz',**m);np.savez_compressed(dest/'reference.npz',**ref)
        audit.append(dict(scenario=name,measurement_sha256=sha(dest/'measurements.npz'),reference_sha256=sha(dest/'reference.npz'),measurement_keys=list(m),samples=len(m['time']),truth_bias_is_not_a_measurement_channel=True))
    np.savez_compressed(ROOT/'audit/measurement_error_draws.npz',**draws)
    stats=[]
    for b in [1,2,3]:
        for key,sd in [(f'noise_acc{b}',cfg['acc_white_noise_sd_m_s2']),(f'noise_gyr{b}',cfg['gyro_white_noise_sd_rad_s'])]:
            x=draws[key];stats.append(dict(channel=key,declared_sample_sd=sd,realized_mean=x.mean(axis=0),realized_sample_sd=x.std(axis=0,ddof=1),normalized_mean=x.mean(axis=0)/(sd/np.sqrt(len(x)))))
    save(ROOT/'audit/GENERATION.json',dict(configuration=cfg,source_hashes={p.name:sha(p) for p in [ROOT/'generate_measurement_errors.py',ROOT/'workflow_io.py',ROOT/'configs/measurement_errors.json']},clean_input_hashes={p.name:sha(p) for p in (ROOT/'data/clean').glob('*.npz')},scenarios=audit,realized_noise_statistics=stats,poses_and_geometry_unchanged=True,no_new_dynamics_simulation=True))
    print('Generated common inputs:',', '.join(scenarios),flush=True)


if __name__=='__main__':main()
