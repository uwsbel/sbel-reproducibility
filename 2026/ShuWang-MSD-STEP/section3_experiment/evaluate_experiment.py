"""Reference comparison after state estimates are locked.

Geometry is required explicitly; no choice is made by minimizing state errors.
The synchronization uses measured angular velocity, never STEP estimates.
"""
from run_case import ROOT,save,sha
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import argparse,json

def main(tag,geometry):
    run=ROOT/'runs'/tag;out=run/('reference_'+geometry);out.mkdir(exist_ok=False)
    settings=json.loads((ROOT/'configs/experimental_reference.json').read_text())
    df=pd.read_csv(ROOT/'data/experimental_optical_angles.csv')
    tv=df.Frame.to_numpy()/settings['fps'];ang=np.unwrap(np.deg2rad(df[['AngleBar1','AngleBar2']].to_numpy()),axis=0);ang-=ang[0]
    time=np.load(ROOT/'data/experimental_measurements.npz')['time']
    offset=settings['offset_video_minus_imu_s']
    qtime=time+offset;mask=(qtime>=tv[0]-1e-12)&(qtime<=tv[-1]+1e-12)
    tgrid=np.arange(tv[0],tv[-1],settings['uniform_dt']);a=CubicSpline(tv,ang)(tgrid)
    refs={}
    length_options={
        'notebook':settings['lengths_m'],
        'optical':[.13,.24,.12],
    }
    lengths=length_options.get(geometry,[.13,.24,.12])
    for b in [1,2]:
        r=R.from_euler('y',a[:,b-1]).as_matrix();refs[f'R{b}']=R.from_euler('y',CubicSpline(tv,ang)(qtime[mask])[:,b-1]).as_matrix()
        local=np.array([0.,0.,-lengths[0 if b==1 else 2]])
        p=np.einsum('nij,j->ni',r,local)
        if b==2:p+=np.einsum('nij,j->ni',R.from_euler('y',a[:,0]).as_matrix(),np.array([0.,0.,-lengths[1]]))
        vel=savgol_filter(p,settings['velocity_sg_window'],settings['velocity_sg_order'],deriv=1,delta=settings['uniform_dt'],axis=0)
        refs[f'p{b}']=CubicSpline(tgrid,p)(qtime[mask]);refs[f'v{b}']=CubicSpline(tgrid,vel)(qtime[mask])
    rows=[];geo=[];fig,ax=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for b in [1,2]:
        h=np.load(run/f'body{b}_states.npz');c=json.loads((run/f'body{b}_stageI.json').read_text());rr=refs[f'R{b}']
        angle=np.rad2deg(R.from_matrix(np.transpose(rr,(0,2,1))@h['R'][mask]).magnitude())
        row=dict(body=b,geometry_option=geometry,samples=int(mask.sum()),position_rmse_m=None,velocity_rmse_m_s=None,attitude_mean_deg=float(angle.mean()),attitude_rmse_deg=float(np.sqrt(np.mean(angle**2))),initial_attitude_deg=None)
        if geometry!='angles_only':
            for k,label in [('p','position_rmse_m'),('v','velocity_rmse_m_s')]:row[label]=float(np.sqrt(np.mean(np.sum((h[k][mask]-refs[f'{k}{b}'])**2,axis=1))))
            for key,ref in [('lever',np.array([0.,0.,lengths[0 if b==1 else 2]])),*([('parent_lever',np.array([0.,0.,-(lengths[1]-lengths[0])]))] if b==2 else [])]:
                est=np.array(c[key]);geo.append(dict(body_stage=b,vector=key,reference=ref,estimate=est,in_plane_error_percent=float(100*np.linalg.norm((est-ref)[[0,2]])/np.linalg.norm(ref[[0,2]]))))
        rows.append(row)
        for j,k in enumerate(['p','v','R']):
            est=h[k][mask] if k!='R' else R.from_matrix(h['R'][mask]).as_quat()
            ref=refs[f'{k}{b}'] if k!='R' else R.from_matrix(rr).as_quat()
            if k=='R':est*=np.where(np.sum(est*ref,axis=1)<0,-1,1)[:,None]
            for d in range(est.shape[1]):
                ax[b-1,j].plot(time[mask],est[:,d],lw=.7)
                if geometry!='angles_only' or k=='R':ax[b-1,j].plot(time[mask],ref[:,d],'--',lw=.7)
            ax[b-1,j].set_title(f'Body {b}: '+['position (m)','velocity (m/s)','quaternion xyzw'][j]);ax[b-1,j].grid(alpha=.2)
            if b==2:ax[b-1,j].set_xlabel('Time (s)')
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);save(out/'geometry.json',geo)
    fig.suptitle('Solid: reconstruction; dashed: optical reference',fontsize=10);fig.tight_layout();fig.savefig(out/'state_histories.png',dpi=170);fig.savefig(out/'state_histories.pdf');plt.close(fig)
    save(out/'PROTOCOL.json',dict(geometry=geometry,lengths_m=None if geometry=='angles_only' else lengths,
        offset_video_minus_imu_s=offset,offset_source='Measured gyroscope/optical angular-velocity cross-correlation over record, not reconstruction-error minimization.',
        fps=settings['fps'],uniform_dt=.005,velocity_sg_window=31,velocity_sg_order=3,
        orientation_reference='Optical angle relative to first frame, rotation about +Y. No fitted per-body spatial alignment.',
        first_two_samples_excluded='Negative optical times after offset; no extrapolation.',
        initial_attitude_error='Not reported: t=0 precedes optical overlap.',
        status='Optical reference: link angles from the camera frames, rotation about +Y; positions from the link lengths of the geometry option.',
        estimate_hashes={f'body{b}_states.npz':sha(run/f'body{b}_states.npz') for b in [1,2]}))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='experimental_v1');p.add_argument('--geometry',choices=['notebook','optical','angles_only'],required=True);a=p.parse_args();main(a.run,a.geometry)
