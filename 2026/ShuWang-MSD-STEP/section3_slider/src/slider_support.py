"""Freeze -> generate separate input/reference files -> estimate -> score.

No numerical settings are changed after seeing reconstruction truth errors.
"""
from __future__ import annotations
import ast
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as step

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')

def test_jacobians():
    rng=np.random.default_rng(20261006);worst={}
    for _ in range(12):
        p,v,ba,bg,w,ell,jp,jv=rng.normal(size=(8,3));rot=R.random(random_state=rng).as_matrix()
        n=step.unit(rng.normal(size=3));a=step.unit(rng.normal(size=3));d=step.unit(rng.normal(size=3));c=rng.normal(size=3)
        ref=rot@R.from_rotvec(rng.normal(size=3)*.2).as_matrix()
        kw=dict(axis_child=n,axis_world=a,guide=(c,d),attitude_ref=ref)
        blocks=step.residuals(p,v,rot,ba,bg,w,ell,jp,jv,**kw)
        eps=1e-7
        for bi,(name,y,H) in enumerate(blocks):
            cols=[]
            for k in range(15):
                dx=np.eye(15)[k]*eps
                vals=[]
                for sign in (1,-1):
                    x=sign*dx
                    z=step.residuals(p+x[:3],v+x[3:6],rot@R.from_rotvec(x[6:9]).as_matrix(),
                        ba+x[9:12],bg+x[12:15],w,ell,jp,jv,**kw)[bi][1]
                    vals.append(z)
                cols.append((vals[0]-vals[1])/(2*eps))
            err=float(np.max(np.abs(np.column_stack(cols)-H)));worst[name]=max(worst.get(name,0.),err)
    assert max(worst.values())<1e-6,worst
    n=step.unit([.2,.4,1.]);r=R.from_rotvec([.1,-.3,.2]);a=r.apply(n);E=step.tangent(a)
    free=max(np.linalg.norm(E.T@(r*R.from_rotvec(n*t)).apply(n)) for t in np.linspace(-3,3,21))
    assert free<1e-12
    return dict(residual_jacobian_max_errors=worst,free_hinge_rotation_residual=float(free))

def get_generator(path):
    src=path.read_text(encoding='utf-8-sig')
    node=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='generate_slider_crank_trajectory_v3')
    namespace=dict(np=np,R=R)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name]

def generate_data(generator,cfg):
    args=dict(L1=2.,L2=4.,omega=2.,duration=20.,dt=.005,noise_std_acc=.01,noise_std_gyr=.001,bias_acc_range=.02,bias_gyr_range=.002)
    noisy=generator(**args)
    clean=generator(**{**args,'noise_std_acc':0.,'noise_std_gyr':0.,'bias_acc_range':0.,'bias_gyr_range':0.})
    ell3=np.array(cfg['slider_lever']);Rp=R.from_quat(noisy['imu3']['quat'])
    # The slider does not rotate, so a rigid mounting offset
    # shifts position, but not true acceleration, velocity, or gyro channels.
    noisy['imu3']['pos']=noisy['imu3']['pos']-Rp.apply(ell3)
    clean['imu3']['pos']=clean['imu3']['pos']-Rp.apply(ell3)
    measures=dict(time=noisy['time']);truth=dict(time=noisy['time'])
    for b in (1,2,3):
        z=noisy[f'imu{b}'];cl=clean[f'imu{b}']
        measures.update({f'acc{b}':z['acc'],f'gyr{b}':z['gyr']})
        rs=np.random.RandomState(b);ba=rs.uniform(-.02,.02,3);bg=rs.uniform(-.002,.002,3)
        truth.update({f'p{b}':z['pos'],f'v{b}':z['vel'],f'R{b}':R.from_quat(z['quat']).as_matrix(),
            f'ba{b}':ba,f'bg{b}':bg,f'acc_clean{b}':cl['acc'],f'gyr_clean{b}':cl['gyr']})
    truth.update(lever_1A=np.array([-1.,0.,0.]),lever_1B=np.array([1.,0.,0.]),
        lever_2B=np.array([-2.,0.,0.]),lever_2C=np.array([2.,0.,0.]),lever_3C=ell3,
        axis=np.array([0.,0.,1.]),guide=np.array([1.,0.,0.]))
    # Verify exact analytic closure and transported velocity, including IMU3 offset.
    closure={}
    for label,i,j,li,lj in [('A',0,1,None,truth['lever_1A']),('B',1,2,truth['lever_1B'],truth['lever_2B']),('C',2,3,truth['lever_2C'],ell3)]:
        pj=truth[f'p{j}']+np.einsum('nij,j->ni',truth[f'R{j}'],lj)
        vj=truth[f'v{j}']+step.apply(truth[f'R{j}'],np.cross(truth[f'gyr_clean{j}'],lj))
        pi=np.zeros_like(pj) if i==0 else truth[f'p{i}']+np.einsum('nij,j->ni',truth[f'R{i}'],li)
        vi=np.zeros_like(vj) if i==0 else truth[f'v{i}']+step.apply(truth[f'R{i}'],np.cross(truth[f'gyr_clean{i}'],li))
        closure[label]=dict(position_max_m=float(np.max(np.abs(pj-pi))),velocity_max_m_s=float(np.max(np.abs(vj-vi))))
        assert max(closure[label].values())<1e-12
    data=HERE/'data';data.mkdir()
    np.savez_compressed(data/'imu_measurements.npz',**measures)
    np.savez_compressed(data/'evaluation_truth.npz',**truth)
    save(data/'GENERATION.json',dict(generator_args=args,seeds=[1,2,3],slider_lever=ell3,
        slider_offset_rule='p_IMU3=p_C-R3*ell3C; constant true attitude implies unchanged acceleration and velocity',
        analytic_closure_checks=closure,measurement_sha256=sha(data/'imu_measurements.npz'),truth_sha256=sha(data/'evaluation_truth.npz')))
    return closure

def evaluate(cals,hists,truth,t):
    out=HERE/'results';rows=[];geo=[];errs={}
    for b in (1,2,3):
        h=hists[b];cal=cals[b];er=np.rad2deg((R.from_matrix(truth[f'R{b}']).inv()*R.from_matrix(h['R'])).magnitude())
        ep=np.linalg.norm(h['p']-truth[f'p{b}'],axis=1);ev=np.linalg.norm(h['v']-truth[f'v{b}'],axis=1)
        ea=np.linalg.norm(h['ba']-truth[f'ba{b}'],axis=1);eg=np.linalg.norm(h['bg']-truth[f'bg{b}'],axis=1)
        nd=cal['axis'] if b<3 else cal['guide_direction']
        ndref=truth['axis'] if b<3 else truth['guide']
        direction_error=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(nd,ndref)),0,1))))
        rows.append(dict(body=b,initial_attitude_error_deg=float(er[0]),position_rmse_m=float(np.sqrt(np.mean(ep**2))),
            velocity_rmse_m_s=float(np.sqrt(np.mean(ev**2))),attitude_mean_deg=float(er.mean()),attitude_rmse_deg=float(np.sqrt(np.mean(er**2))),
            accel_bias_rmse_m_s2=float(np.sqrt(np.mean(ea**2))),gyro_bias_rmse_rad_s=float(np.sqrt(np.mean(eg**2))),
            accel_bias_final_error_m_s2=float(ea[-1]),gyro_bias_final_error_rad_s=float(eg[-1]),
            direction_error_deg=direction_error))
        errs[b]=(ep,ev,er)
        pd.DataFrame(dict(time_s=t,position_error_m=ep,velocity_error_m_s=ev,attitude_error_deg=er,
            accel_bias_error_m_s2=ea,gyro_bias_error_rad_s=eg)).to_csv(out/f'body{b}_errors.csv',index=False)
    for label,estimate in [('1A',cals[1]['lever']),('1B',cals[2]['parent_lever']),('2B',cals[2]['lever']),('2C',cals[3]['parent_lever']),('3C',cals[3]['lever'])]:
        gt=truth['lever_'+label]
        geo.append(dict(vector=label,estimated=label!='3C',estimate_m=estimate,ground_truth_m=gt,
            in_plane_relative_error_percent=None if label=='3C' else float(100*np.linalg.norm(estimate[:2]-gt[:2])/np.linalg.norm(gt[:2])),
            axial_component_m=float(estimate[2]),full_vector_difference_m=float(np.linalg.norm(estimate-gt))))
    save(out/'state_metrics.json',rows);pd.DataFrame(rows).to_csv(out/'state_metrics.csv',index=False)
    save(out/'geometry_metrics.json',geo)
    pd.DataFrame([{**{k:v for k,v in x.items() if k not in ('estimate_m','ground_truth_m')},
        **{f'estimated_{c}_m':x['estimate_m'][k] for k,c in enumerate('xyz')}} for x in geo]).to_csv(out/'geometry_metrics.csv',index=False)
    fig,axs=plt.subplots(3,3,figsize=(12,8),sharex=True)
    for b in (1,2,3):
        for j,(title,unit) in enumerate([('Position error','m'),('Velocity error','m/s'),('Attitude error','deg')]):
            ax=axs[b-1,j];ax.plot(t,errs[b][j],lw=1,color='#0072B2');ax.set_title(f'Body {b}: {title}');ax.set_ylabel(unit);ax.grid(alpha=.2)
            if b==3:ax.set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'full_pipeline_errors.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(12,8),sharex=True)
    colors=['#0072B2','#D55E00','#009E73']
    for b in (1,2,3):
        h=hists[b]
        for j,key in enumerate(('p','v','ba')):
            target=truth[f'{key}{b}']
            if target.ndim==1:target=np.tile(target,(len(t),1))
            for k,color in enumerate(colors):
                axs[b-1,j].plot(t,h[key][:,k],color=color,lw=1,label='xyz'[k]+' estimate')
                axs[b-1,j].plot(t,target[:,k],color=color,lw=.8,ls='--',label='xyz'[k]+' truth')
            axs[b-1,j].set_title(f'Body {b}: '+['position (m)','velocity (m/s)','accelerometer bias (m/s2)'][j]);axs[b-1,j].grid(alpha=.2)
            if b==3:axs[b-1,j].set_xlabel('Time (s)')
    axs[0,0].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'state_and_bias_traces.png',dpi=180);plt.close(fig)
    return rows,geo
