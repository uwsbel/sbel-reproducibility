"""Run one measurement-error scenario: STEP, then JT from the reference initial states, scoring and timing."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import sys,time,json,argparse,csv,shutil,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy
import calibration,core,compiled_step,jt_checked as jt
from workflow_io import save,sha


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def chart(h,G,pivot):
    return dict(p=(h['p']-pivot)@G.T+pivot,v=h['v']@G.T,R=G@h['R'])


def metrics(h,ref,b,mask,method,level,frame):
    dp=np.linalg.norm(h['p']-ref[f'p{b}'],axis=1);dv=np.linalg.norm(h['v']-ref[f'v{b}'],axis=1)
    angle=np.rad2deg(R.from_matrix(np.transpose(ref[f'R{b}'],(0,2,1))@h['R']).magnitude())
    return dict(method=method,perturbation_component_percent=level,perturbation_norm_percent='' if level=='' else abs(level)*np.sqrt(3),body=b,frame=frame,sample_count=int(mask.sum()),position_rmse_m=float(np.sqrt(np.mean(dp[mask]**2))),velocity_rmse_m_s=float(np.sqrt(np.mean(dv[mask]**2))),attitude_mean_deg=float(angle[mask].mean()),attitude_rmse_deg=float(np.sqrt(np.mean(angle[mask]**2))))


def score(out,cfg,m,ref,step,lever,alljt):
    mask=m['time']>=cfg['exclude_s'];pivot=np.array([0.,0.,6.]);_,G=calibration.gauge(ref['R1'][0],np.array([0.,0.,-9.81]));target={}
    for b in [1,2,3]:
        h=chart(dict(p=ref[f'p{b}'],v=ref[f'v{b}'],R=ref[f'R{b}']),G,pivot)
        target.update({f'p{b}':h['p'],f'v{b}':h['v'],f'R{b}':h['R']})
    rows=[];geos=[];inverse_differences=[]
    def append_both(h,b,method,level):
        primary=metrics(h,target,b,mask,method,level,'declared_heading_chart');rows.append(primary)
        secondary=metrics(chart(h,G.T,pivot),ref,b,mask,method,level,'raw_world_inverse_chart');rows.append(secondary)
        inverse_differences.append(max(abs(primary[k]-secondary[k]) for k in ['position_rmse_m','velocity_rmse_m_s','attitude_mean_deg','attitude_rmse_deg']))
    labels=['L1p','L1d','L2p','L2d','L3p'];truth=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.]])
    for b in [1,2,3]:append_both(step[b],b,'STEP','')
    estimates=[lever[1]['lever'],lever[2]['parent_lever'],lever[2]['lever'],lever[3]['parent_lever'],lever[3]['lever']]
    for label,x,gt in zip(labels,estimates,truth):geos.append(dict(method='STEP',perturbation_component_percent='',vector=label,error_percent=float(100*np.linalg.norm(x-gt)/np.linalg.norm(gt)),metric_time='Stage I locked estimate'))
    numerical=[]
    for level,h in alljt.items():
        for b in [1,2,3]:append_both(dict(p=h[f'r{b}'],v=h[f'v{b}'],R=R.from_quat(h[f'q{b}'][:,[1,2,3,0]]).as_matrix()),b,'JT',level)
        for label,gt in zip(labels,truth):geos.append(dict(method='JT',perturbation_component_percent=level,vector=label,error_percent=float(100*np.linalg.norm(h[label][-1]-gt)/np.linalg.norm(gt)),metric_time='final sample'))
        d=h['numerical_checks'];numerical.append(dict(level=level,covariance_min_eigenvalue=float(d[:,1].min()),covariance_max_asymmetry=float(d[:,2].max()),innovation_max_condition=float(d[:,3].max()),sampled_correction_max_rad=float(d[:,4].max()),maximum_sampled_lever_norm_m=float(d[:,5].max()),diagnostic_stride=100,all_finite=bool(all(np.isfinite(x).all() for x in h.values()))))
    write_csv(out/'state_metrics.csv',rows);write_csv(out/'geometry_metrics.csv',geos);save(out/'JT_NUMERICAL_CHECKS.json',numerical)
    save(out/'EVALUATION_COORDINATES.json',dict(G=G,pivot=pivot,primary=cfg['primary_frame'],secondary='Inverse of the same prescribed chart for BOTH estimates and reference; not an alternative direct raw score.',max_metric_difference_under_common_frame_transform=max(inverse_differences),no_per_body_alignment=True,no_fitted_trajectory_alignment=True))
    return rows,geos,numerical


def run(scenario,tag,nominal_only=False,skip_timing=False):
    out=ROOT/'runs'/tag;out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/comparison_noise_bias.json').read_text(encoding='utf-8'));scfg=json.loads((ROOT/'configs'/os.environ.get('STEP_STAGE_CONFIG','step.json')).read_text(encoding='utf-8'))
    cfg['active_scenario']=scenario
    cfg['measurement_errors']=json.loads((ROOT/'configs/measurement_errors.json').read_text())
    if nominal_only:cfg['perturbation_component_percent']=[0]
    cfg['nominal_only']=nominal_only;cfg['skip_timing']=skip_timing
    files=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'configs').glob('*.json'))+[Path(__file__),ROOT/'workflow_io.py',ROOT/'generate_measurement_errors.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        dest=out/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'FROZEN_PROTOCOL.json',dict(configuration=cfg,step_configuration=scfg,source_hashes=hashes,measurement_error_generation_sha256=sha(ROOT/'audit/GENERATION.json'),input_hashes={p.name:sha(p) for p in (ROOT/'data'/scenario).glob('*.npz')},software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),truth_policy='STEP runs and saves its estimates (with SHA-256 hashes) before the reference is opened. The reference then supplies only the JT initial states and the scoring.'))
    with np.load(ROOT/'data'/scenario/'measurements.npz') as z:m={k:z[k] for k in z.files}
    measurement_sha=sha(ROOT/'data'/scenario/'measurements.npz')
    save(out/'METHOD_INPUT_HASHES.json',dict(STEP=measurement_sha,JT=measurement_sha,no_oracle_bias_subtraction=True))
    dt=float(np.median(np.diff(m['time'])));nc=int(round(scfg['calibration_s']/dt));cals={};step={};stage_timings=[]
    if 'stage1_gyro_range_rad_s' in scfg:
        calibration.STAGE1_KEEP=calibration.range_keep([m[f'gyr{b}'][:nc] for b in (1,2,3)],scfg['stage1_gyro_range_rad_s'],scfg['stage1_exclusion_half_width'])
        save(out/'STAGE1_RANGE_RULE.json',dict(excluded=int((~calibration.STAGE1_KEEP).sum()),samples=int(nc),limit_rad_s=scfg['stage1_gyro_range_rad_s'],half_width=scfg['stage1_exclusion_half_width']))
    for b in [1,2,3]:
        print('STEP body',b,'Stage I',flush=True);tic=time.perf_counter();parent=step.get(b-1)
        cal=calibration.calibrate(b,m[f'acc{b}'][:nc],m[f'gyr{b}'][:nc],dt,scfg,parent)
        ti=time.perf_counter()-tic;save(out/f'body{b}_stageI.json',cal)
        print('STEP body',b,'Stage II reference implementation',flush=True);tic=time.perf_counter()
        h=core.reconstruct(b,m[f'acc{b}'],m[f'gyr{b}'],dt,cal,scfg,parent)
        stage_timings.append(dict(body=b,stageI_s=ti,reference_stageII_s=time.perf_counter()-tic))
        cals[b]=cal;step[b]=h;save(out/f'body{b}_numerical_checks.json',h['checks'])
        np.savez_compressed(out/f'body{b}_states.npz',time=m['time'],**{k:v for k,v in h.items() if isinstance(v,np.ndarray)})
    save(out/'STEP_PRE_REFERENCE_HASHES.json',{p.name:sha(p) for p in out.glob('body*') if p.is_file()})
    save(out/'STEP_STAGE_TIMING.json',stage_timings)
    print('STEP estimates locked; opening reference for JT initialization and scoring.',flush=True)
    with np.load(ROOT/'data'/scenario/'reference.npz') as z:ref={k:z[k] for k in z.files}
    pivot=np.array(scfg['fixed_support']);g=np.array(scfg['gravity']);_,G=calibration.gauge(ref['R1'][0],g);states={}
    for b in [1,2,3]:
        states.update({f'r{b}':G@(ref[f'p{b}'][0]-pivot)+pivot,f'v{b}':G@ref[f'v{b}'][0],f'q{b}':R.from_matrix(G@ref[f'R{b}'][0]).as_quat()[[3,0,1,2]]})
    labels=['L1p','L1d','L2p','L2d','L3p'];truth=np.array([[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.]])
    signs=np.array(cfg['perturbation_signs']);kw={k:m[k] for k in ['acc1','gyr1','acc2','gyr2','acc3','gyr3']}
    kw.update(dt=dt,g_vec=g,pos_fixed_SJ=pivot,init_states=states,R_diag=cfg['jt_constraint_sd']['position_m'],Qd_diag=cfg['jt_Qd_diagonal'],n_iters=1)
    print('JT diagnostic warm-up',flush=True);small={**kw,**{k:v[:10] for k,v in kw.items() if k.startswith(('acc','gyr'))}}
    jt.run_jointtracker_3link_numba_wrapper(**small,init_params=dict(zip(labels,truth.copy())))
    alljt={}
    for level in cfg['perturbation_component_percent']:
        h=jt.run_jointtracker_3link_numba_wrapper(**kw,init_params=dict(zip(labels,truth+level/100*signs)))
        alljt[level]=h;np.savez_compressed(out/f'jt_{level:+d}.npz',time=m['time'],**h)
        print('JT condition',level,'complete',flush=True)
    save(out/'PRE_SCORING_HASHES.json',{p.name:sha(p) for p in out.glob('*.npz')})
    rows,geos,checks=score(out,cfg,m,ref,step,cals,alljt)
    score_bias(out,m,ref,step)
    if skip_timing:
        save(out/'STATUS.json',dict(complete=True,numerical_checks_passed=all(x['all_finite'] and x['covariance_min_eigenvalue']>-1e-10 for x in checks) and all(c['fit']['converged'] for c in cals.values()),all_conditions_computed=not nominal_only,noise_bias_tested=scenario=='noise_bias',manuscript_updated=False,method_superiority_certified=False,replay_nominal_only=True,timing_skipped=True))
        print('Replay complete',out,flush=True)
        return
    print('Verifying compiled STEP equivalence before timing',flush=True);tic=time.perf_counter();hc=compiled_step.run_chain(m,cals,scfg);warm=time.perf_counter()-tic
    diffs={f'body{b}_{k}':float(np.max(np.abs(x-step[b][k]))) for b,h in hc.items() for k,x in h.items()};equivalent=max(diffs.values())<1e-7
    save(out/'STEP_COMPILED_EQUIVALENCE.json',dict(passed=equivalent,threshold=1e-7,differences=diffs))
    if not equivalent:raise RuntimeError('Compiled STEP differs; timing comparison stopped.')
    print('Verifying JT timing path equivalence',flush=True);tic=time.perf_counter();ht=jt.run_jointtracker_3link_numba_wrapper(**kw,init_params=dict(zip(labels,truth.copy())),collect_diagnostics=False);jt_warm=time.perf_counter()-tic
    diffs_jt={k:float(np.max(np.abs(ht[k]-alljt[0][k]))) for k in ht if k!='numerical_checks'};jeq=max(diffs_jt.values())<1e-7
    save(out/'JT_TIMING_EQUIVALENCE.json',dict(passed=jeq,threshold=1e-7,differences=diffs_jt))
    if not jeq:raise RuntimeError('JT timing path differs; timing comparison stopped.')
    # Repeated timing is serial and alternated on one process; diagnostics and all archive I/O are outside timing.
    st=[];jtimes=[]
    for i in range(cfg['timing_repetitions']):
        tic=time.perf_counter();compiled_step.run_chain(m,cals,scfg);st.append(time.perf_counter()-tic)
        tic=time.perf_counter();jt.run_jointtracker_3link_numba_wrapper(**kw,init_params=dict(zip(labels,truth.copy())),collect_diagnostics=False);jtimes.append(time.perf_counter()-tic)
        print('Timing pair',i+1,'complete',flush=True)
    def timing(x):return dict(times_s=x,mean_ms_per_sample=float(np.mean(x)/len(m['time'])*1000),median_ms_per_sample=float(np.median(x)/len(m['time'])*1000),min_ms_per_sample=float(np.min(x)/len(m['time'])*1000),max_ms_per_sample=float(np.max(x)/len(m['time'])*1000))
    save(out/'TIMING_COMPARISON.json',dict(STEP_StageI_total_s=sum(x['stageI_s'] for x in stage_timings),STEP_StageI_by_body=stage_timings,STEP_compiled_StageII=timing(st),JT_zero_perturbation=timing(jtimes),STEP_warmup_s=warm,JT_timing_path_warmup_s=jt_warm,samples=len(m['time']),scope=cfg['timing_scope'],warmup_excluded=True,StageI_included_in_StageII=False,three_bodies_only=True,no_scale_test=True))
    passed=all(x['all_finite'] and x['covariance_min_eigenvalue']>-1e-10 for x in checks) and all(c['fit']['converged'] for c in cals.values())
    save(out/'STATUS.json',dict(complete=True,numerical_checks_passed=passed,all_conditions_computed=True,noise_bias_tested=scenario=='noise_bias',manuscript_updated=False,method_superiority_certified=False))
    print('Complete',out,flush=True)


def score_bias(out,m,ref,step):
    mask=m['time']>=3.;rows=[]
    for b in [1,2,3]:
        for key,unit in [('ba','m/s^2'),('bg','rad/s')]:
            truth=ref[f'{key}{b}'];est=step[b][key];error=est[mask]-truth
            rows.append(dict(method='STEP',body=b,bias=key,unit=unit,truth_components=truth,estimated_final_components=est[-1],estimated_mean_components=est[mask].mean(axis=0),bias_vector_rmse=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),per_component_rmse=np.sqrt(np.mean(error**2,axis=0))))
    save(out/'STEP_BIAS_METRICS.json',rows)
    save(out/'BIAS_POLICY.json',dict(input_correction='Neither method receives the true biases or measurements corrected with them',STEP='Stage I has no bias parameters; Stage II initializes ba/bg at zero and estimates them.',JT='42-dimensional implementation has no sensor bias states. It processes the same uncorrected input.',scope='Bias-case results assess this complete workflow and these differing bias capabilities, not estimator structure in isolation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scenario',choices=['noise_only','bias_only','noise_bias','clean']);p.add_argument('--tag',required=True);p.add_argument('--nominal-only',action='store_true');p.add_argument('--skip-timing',action='store_true');a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run tag')
    run(a.scenario,a.tag,a.nominal_only,a.skip_timing)
