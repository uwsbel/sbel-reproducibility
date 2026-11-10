"""Run cases separately, with immutable run directories and delayed scoring."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,platform,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA_DIR=Path(os.environ.get('STEP_DATA_DIR', str(ROOT/'data'))).resolve()
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core,calibration

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else bool(v) if isinstance(v,np.bool_) else int(v) if isinstance(v,np.integer) else str(v)),encoding='utf-8')
def freeze(out,cfg):
    if out.parent.resolve() != (ROOT/'runs').resolve() or not out.name.replace('_','').replace('-','').isalnum():
        raise ValueError('Run tag must contain only letters, numbers, hyphens or underscores')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT/'src').glob('*.py')) + sorted((ROOT/'configs').glob('*.json'))
    hashes = {}
    for p in files:
        rel = p.relative_to(ROOT); hashes[rel.as_posix()] = sha(p)
        dest = out/'code_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    save(out/'FROZEN_PROTOCOL.json',dict(configuration=cfg,source_hashes=hashes,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        truth_access='Estimation accepts measurement channels and declared boundary inputs. Scoring opens reference files only after estimates are saved and hashed.',
        gauge='Estimated anchor fixes a coordinate azimuth; no true attitude initializes STEP.',
        selection='Minimum finite measurement residual across fixed starts; convergence checked separately.',
        numerical_settings_origin='Record length 20 s: Stage I 10 s, Stage II 20 s. Integration and noise are generated in fixed 10 s blocks, so the first 10 s of the record do not depend on its length.'))


def prepare_spatial(cfg):
    path=DATA_DIR/'spatial_measurements.npz'
    if path.exists():return
    from generate_spatial import run_simulation_best_practice
    df,checks=run_simulation_best_practice(duration=cfg['simulation_duration_s'])
    m=dict(time=df.Time_s.to_numpy());truth=dict(time=m['time']);truth['support']=np.array(cfg['fixed_support'])
    for b in [1,2]:
        rng=np.random.default_rng(20261006+b)
        ba=rng.uniform(-.02,.02,3);bg=rng.uniform(-.002,.002,3)
        a=df[[f'IMU{b}_Acc_{c}' for c in 'XYZ']].to_numpy();w=df[[f'IMU{b}_Gyro_{c}' for c in 'XYZ']].to_numpy()
        # Fixed-size noise blocks: the first 10 s of noise do not depend on the record length.
        an=np.empty_like(a);wn=np.empty_like(w)
        block=int(round(cfg['generation_block_s']/cfg['simulation_dt_s']))
        for start in range(0,len(a),block):
            end=min(start+block,len(a));shape=(end-start,3)
            an[start:end]=rng.normal(0,.01,shape)
            wn[start:end]=rng.normal(0,.001,shape)
        m[f'acc{b}']=a+ba+an;m[f'gyr{b}']=w+bg+wn
        for k,prefix in [('p','Pos'),('v','Vel')]:truth[f'{k}{b}']=df[[f'IMU{b}_{prefix}_{c}' for c in 'XYZ']].to_numpy()
        truth[f'R{b}']=R.from_quat(df[[f'IMU{b}_Quat_{c}' for c in 'XYZW']].to_numpy()).as_matrix()
        truth[f'ba{b}']=ba;truth[f'bg{b}']=bg
    truth.update(lever1=np.array([0.,0.,1.]),lever2=np.array([0.,0.,1.]),parent_lever2=np.array([0.,0.,-1.]))
    np.savez_compressed(path,**m);np.savez_compressed(DATA_DIR/'spatial_reference.npz',**truth)
    save(DATA_DIR/'spatial_generation.json',dict(checks=checks,noise_seed=[20261007,20261008],noise_sd=[.01,.001],bias_bounds=[.02,.002],duration_s=cfg['simulation_duration_s'],noise_block_s=cfg['generation_block_s'],origin='Spatial double-pendulum simulation, 20 s; integration and noise draws in fixed 10 s blocks, so the first 10 s do not depend on the record length.',measurement_sha256=sha(path)))

def score(out,cfg,cals,hists,m):
    reference_path=DATA_DIR/f"{cfg['case']}_reference.npz"
    save(out/'REFERENCE_HASH.json',dict(file=reference_path.name,sha256=sha(reference_path)))
    ref=np.load(reference_path)
    rows=[];geos=[];nb=cfg['n_bodies'];start=int(round(cfg.get('exclude_s',0)/(m['time'][1]-m['time'][0])))
    # Only evaluator accesses reference attitude, to express it in the declared chart.
    if cfg['joint_types'][0]=='spherical':
        _,G=calibration.gauge(ref['R1'][0],cfg['gravity'])
    else:
        _,G=calibration.gauge(ref['R1'][0],cfg['gravity'],ref['axis1'],cfg['heading_axis_target'])
    save(out/'EVALUATION_COORDINATE_TRANSFORM.json',dict(reference_to_estimator_chart=G,scope='One common rotation about gravity for all reference states about the prescribed anchor. Applied after estimates are locked; no per-body alignment. Raw errors also saved.'))
    fig,axes=plt.subplots(nb,3,figsize=(12,3*nb),squeeze=False,sharex=True)
    for b in range(1,nb+1):
        h=hists[b];p=ref[f'p{b}'];v=ref[f'v{b}'];rot=ref[f'R{b}'];pivot=np.array(cfg['fixed_support'])
        pg=(p-pivot)@G.T+pivot;vg=v@G.T;rg=G@rot
        for frame,pp,vv,rr in [('raw_world',p,v,rot),('declared_heading_chart',pg,vg,rg)]:
            dp=np.linalg.norm(h['p']-pp,axis=1);dv=np.linalg.norm(h['v']-vv,axis=1)
            da=np.rad2deg(R.from_matrix(np.transpose(rr,(0,2,1))@h['R']).magnitude())
            a0=float(np.rad2deg(R.from_matrix(rr[0].T@cals[b]['R0']).magnitude()))
            rows.append(dict(body=b,frame=frame,initial_attitude_deg=a0,position_rmse_m=float(np.sqrt(np.mean(dp[start:]**2))),velocity_rmse_m_s=float(np.sqrt(np.mean(dv[start:]**2))),attitude_mean_deg=float(np.mean(da[start:])),attitude_rmse_deg=float(np.sqrt(np.mean(da[start:]**2))),sample_count=len(dp)-start))
        for j,(k,target) in enumerate([('p',pg),('v',vg),('R',rg)]):
            est=h[k] if k!='R' else R.from_matrix(h[k]).as_quat()
            tar=target if k!='R' else R.from_matrix(target).as_quat()
            if k=='R':est=est*np.where(np.sum(est*tar,axis=1)<0,-1,1)[:,None]
            for q in range(est.shape[1]):
                axes[b-1,j].plot(m['time'],est[:,q],lw=.8)
                axes[b-1,j].plot(m['time'],tar[:,q],'--',lw=.7)
            axes[b-1,j].set_title(f'Body {b}: '+['position (m)','velocity (m/s)','quaternion xyzw'][j]);axes[b-1,j].grid(alpha=.2)
            if b==nb:axes[b-1,j].set_xlabel('Time (s)')
        for name in ['lever','parent_lever']:
            if name not in cals[b]:continue
            gt=ref[f'{name}{b}'];est=cals[b][name];idx=cfg.get('geometry_components',[0,1,2])
            geos.append(dict(body_stage=b,vector=name,estimate_m=est,reference_m=gt,error_percent=float(100*np.linalg.norm((est-gt)[idx])/np.linalg.norm(gt[idx]))))
    fig.suptitle('Solid: reconstruction; dashed: reference in declared heading chart',fontsize=10);fig.tight_layout()
    fig.savefig(out/'state_histories.png',dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'state_metrics.csv',index=False);save(out/'geometry_metrics.json',geos)
    save(out/'metrics.json',rows)
    return rows,geos

def run(case,tag):
    global DATA_DIR
    cfg=json.loads((ROOT/'configs'/os.environ.get('STEP_CASE_CONFIG',f'{case}.json')).read_text());out=ROOT/'runs'/tag;freeze(out,cfg)
    DATA_DIR=out/'data';DATA_DIR.mkdir()
    src_data=os.environ.get('STEP_COPY_DATA_FROM')
    if src_data:
        for f in Path(src_data).glob('*'):shutil.copy2(f,DATA_DIR/f.name)
    print('Frozen',out,flush=True)
    if case=='spatial':prepare_spatial(cfg)
    m0=np.load(DATA_DIR/f'{case}_measurements.npz');m={k:m0[k] for k in m0.files};m0.close()
    save(out/'INPUT_HASHES.json',{f'{case}_measurements.npz':sha(DATA_DIR/f'{case}_measurements.npz')})
    dt=float(np.median(np.diff(m['time'])));nc=int(round(cfg['calibration_s']/dt));cals={};hists={};timings=[]
    if 'stage1_gyro_range_rad_s' in cfg:
        calibration.STAGE1_KEEP=calibration.range_keep([m[f'gyr{b}'][:nc] for b in range(1,cfg['n_bodies']+1)],cfg['stage1_gyro_range_rad_s'],cfg['stage1_exclusion_half_width'])
        save(out/'STAGE1_RANGE_RULE.json',dict(excluded=int((~calibration.STAGE1_KEEP).sum()),samples=int(nc),limit_rad_s=cfg['stage1_gyro_range_rad_s'],half_width=cfg['stage1_exclusion_half_width']))
    for b in range(1,cfg['n_bodies']+1):
        print(f'Body {b}: Stage I ({nc} samples)',flush=True);tic=time.perf_counter();parent=hists.get(b-1)
        cal=calibration.calibrate(b,m[f'acc{b}'][:nc],m[f'gyr{b}'][:nc],dt,cfg,parent)
        ti=time.perf_counter()-tic;save(out/f'body{b}_stageI.json',cal)
        print(f"Stage I cost {cal['fit']['cost']:.6g}; converged {cal['fit']['converged']}. Stage II",flush=True);tic=time.perf_counter()
        hist=core.reconstruct(b,m[f'acc{b}'],m[f'gyr{b}'],dt,cal,cfg,parent,cals[1].get('axis') if b>1 else None)
        timings.append(dict(body=b,stageI_s=ti,stageII_s=time.perf_counter()-tic));cals[b]=cal;hists[b]=hist
        np.savez_compressed(out/f'body{b}_states.npz',time=m['time'],**{k:v for k,v in hist.items() if isinstance(v,np.ndarray)})
        save(out/f'body{b}_checks.json',hist['checks'])
        assert np.array_equal(hist['R'][0],cal['R0']) and np.array_equal(hist['p'][0],cal['p0']) and np.array_equal(hist['v'][0],cal['v0'])
    save(out/'timing.json',timings)
    save(out/'PRE_EVALUATION_HASHES.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print('Estimates saved and hashed. Opening reference for evaluation.',flush=True)
    if (DATA_DIR/f'{case}_reference.npz').exists():
        rows,geo=score(out,cfg,cals,hists,m);print(pd.DataFrame(rows).to_string(index=False),flush=True)
    else: rows=[];geo=[]
    save(out/'STATUS.json',dict(computation_complete=True,all_optimizers_converged=all(c['fit']['converged'] for c in cals.values()),reference_scored=bool(rows),accuracy_certified=False))
    print('Complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['spatial']);p.add_argument('--tag',required=True);a=p.parse_args();run(a.case,a.tag)
