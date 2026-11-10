"""Sequential STEP implementation: batch geometry, then local 15-state MEKF.

Estimator module accepts IMU measurements and declared public boundary/mounting
inputs only. It has no truth-file reader or scoring/selection against truth.
Rotations are IMU -> world; error rotations multiply on the right.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.linalg import block_diag


def skew(x):
    x,y,z=x
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def unit(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all() or np.linalg.norm(x)<1e-12:
        raise ValueError('Invalid direction')
    return x/np.linalg.norm(x)

def tangent(n):
    n=unit(n);u=unit(np.cross(n,np.eye(3)[np.argmin(np.abs(n))]))
    return np.column_stack((u,np.cross(n,u)))

def axis_uv(uv):
    return Rot.from_rotvec([uv[0],uv[1],0.]).apply([0.,0.,1.])

def smooth_inputs(acc,gyr,dt,window):
    f=savgol_filter(acc,window,3,axis=0)
    w=savgol_filter(gyr,window,3,axis=0)
    alpha=savgol_filter(gyr,window,3,deriv=1,delta=dt,axis=0)
    A=np.array([skew(a)+skew(v)@skew(v) for a,v in zip(alpha,w)])
    return f,w,A

def increments(gyr,dt):
    """Paper Eq gyroint: M[k+1] = M[k] Exp(w[k] dt)."""
    M=np.empty((len(gyr),3,3));r=Rot.identity();M[0]=np.eye(3)
    for k in range(1,len(gyr)):
        r=r*Rot.from_rotvec(gyr[k-1]*dt);M[k]=r.as_matrix()
    return M

def apply(R,v):
    return np.einsum('nij,nj->ni',R,v)

def initial_guesses(acc,g):
    # A data-derived optimizer guess only, not an assumption of a static record.
    u=unit(np.mean(acc[:20],axis=0));v=unit(-g)
    cr=np.cross(u,v);ang=np.arctan2(np.linalg.norm(cr),np.dot(u,v))
    rg=Rot.from_rotvec(unit(cr)*ang) if np.linalg.norm(cr)>1e-12 else Rot.identity()
    return [np.zeros(3),rg.as_rotvec()]

def fit_best(fun,starts,max_nfev,tol):
    trials=[];sols=[]
    for k,x in enumerate(starts):
        t=time.perf_counter()
        sol=least_squares(fun,x,method='lm',max_nfev=max_nfev,ftol=tol,xtol=tol,gtol=tol)
        finite=np.isfinite(sol.x).all() and np.isfinite(sol.fun).all()
        trials.append(dict(start_index=k,initial_parameters=x,cost=float(sol.cost),
            nfev=int(sol.nfev),status=int(sol.status),success=bool(sol.success),
            message=sol.message,finite=bool(finite),elapsed_s=time.perf_counter()-t))
        sols.append(sol)
    eligible=[k for k,s in enumerate(sols) if np.isfinite(s.x).all() and np.isfinite(s.fun).all()]
    if not eligible:raise RuntimeError('No finite calibration solution')
    # Retain minimum residual independent of optimizer success, but expose failures.
    best=min(eligible,key=lambda k:sols[k].cost);s=sols[best]
    singular=np.linalg.svd(s.jac,compute_uv=False)
    return s.x,dict(selected_start=best,trials=trials,cost=float(s.cost),
        converged=bool(s.success),jacobian_singular_values=singular,
        jacobian_rank_relative_1e_6=int(np.sum(singular>singular[0]*1e-6)))

def heading_gauge(R0,n,g):
    """Choose azimuth of estimated hinge axis, without reading reference attitude.

    World gravity is -Y in this case. Rotate the estimated world hinge axis about
    Y so its horizontal projection lies along +Z. This fixes a coordinate gauge,
    not an externally observed absolute heading.
    """
    assert np.allclose(unit(g),[0.,-1.,0.])
    nw=R0@n
    if np.linalg.norm(nw[[0,2]])<1e-8:raise ValueError('Heading gauge singular')
    beta=-np.arctan2(nw[0],nw[2]);G=Rot.from_rotvec([0.,beta,0.]).as_matrix()
    return G@R0,G

def calibrate(body,acc,gyr,dt,cfg,parent=None,known_child_lever=None):
    g=np.array(cfg['gravity']);window=cfg['sg_windows'][body-1]
    f,w,A=smooth_inputs(acc,gyr,dt,window)
    # Integrate measured gyro as written in the paper; smoothing is for residuals.
    M=increments(gyr,dt)
    bases=initial_guesses(acc,g)
    if body==1:
        weight=cfg['axis_residual_multipliers'][0]
        def fun(x):
            R=Rot.from_rotvec(x[:3]).as_matrix()@M
            ra=apply(R,f+np.einsum('nij,j->ni',A,x[3:6]))+g
            n=axis_uv(x[6:8]);rw=w-(w@n)[:,None]*n
            return np.r_[ra.ravel(),weight*rw.ravel()]
        starts=[np.r_[r,[.1,0.,0.],[0.,0.]] for r in bases]
        x,info=fit_best(fun,starts,cfg['max_nfev'][0],cfg['solver_tolerance'])
        n=axis_uv(x[6:8]);R0,G=heading_gauge(Rot.from_rotvec(x[:3]).as_matrix(),n,g)
        ell=x[3:6]
        return dict(R0=R0,lever=ell,axis=n,p0=-R0@ell,
            v0=-R0@np.cross(gyr[0],ell),gauge_rotation=G,fit=info)
    Rp=parent['R'][:len(acc)]
    # Parent bias states are available; child bias remains unestimated in Stage I.
    fp,wp,Ap=smooth_inputs(parent['acc'][:len(acc)]-parent['ba'][:len(acc)],
        parent['gyr'][:len(acc)]-parent['bg'][:len(acc)],dt,window)
    def fun(x):
        R=Rot.from_rotvec(x[:3]).as_matrix()@M
        ellp=x[3:6];ellj=x[6:9] if body==2 else known_child_lever
        ra=apply(R,f+np.einsum('nij,j->ni',A,ellj))-apply(Rp,fp+np.einsum('nij,j->ni',Ap,ellp))
        if body==3:return ra.ravel()
        n=axis_uv(x[9:11])
        wpw=apply(Rp,wp);wrel=w-apply(np.transpose(R,(0,2,1)),wpw)
        rw=wrel-(wrel@n)[:,None]*n
        return np.r_[ra.ravel(),cfg['axis_residual_multipliers'][1]*rw.ravel()]
    if body==2:
        starts=[np.r_[r,[0.,0.,-.1],[0.,0.,.1],[0.,0.]] for r in bases]
    else:
        starts=[np.r_[(Rot.from_rotvec(unit(g)*a)*Rot.from_rotvec(bases[1])).as_rotvec(),[.1,0.,0.]]
                for a in np.deg2rad([0.,90.,180.,270.])]
    x,info=fit_best(fun,starts,cfg['max_nfev'][body-1],cfg['solver_tolerance'])
    R0=Rot.from_rotvec(x[:3]).as_matrix();ellp=x[3:6]
    ellj=x[6:9] if body==2 else np.array(known_child_lever)
    pj=parent['p'][0]+Rp[0]@ellp
    vj=parent['v'][0]+Rp[0]@np.cross(parent['gyr'][0]-parent['bg'][0],ellp)
    result=dict(R0=R0,parent_lever=ellp,lever=ellj,p0=pj-R0@ellj,
        v0=vj-R0@np.cross(gyr[0],ellj),fit=info)
    if body==2:result['axis']=axis_uv(x[9:11])
    else:
        joints=parent['p'][:len(acc)]+np.einsum('nij,j->ni',Rp,ellp)
        # c is prescribed in this case: fit direction through c, no centroid fit.
        c=np.array(cfg['guide_point']);X=joints-c
        _,sing,V=np.linalg.svd(X,full_matrices=False);d=V[0]
        if d[np.argmax(np.abs(d))]<0:d=-d
        result.update(guide_direction=d,guide_point=c,guide_fit_singular_values=sing,
            guide_perpendicular_rms=float(np.sqrt(np.mean(np.sum((X-(X@d)[:,None]*d)**2,axis=1)))))
    return result

def residuals(p,v,R,ba,bg,w,ell,jp,jv,axis_child=None,axis_world=None,
              guide=None,attitude_ref=None):
    """Returns residual/Jacobian blocks before any covariance weights."""
    u=np.cross(w-bg,ell)
    hp=p+R@ell-jp;hv=v+R@u-jv
    Hp=np.zeros((3,15));Hp[:,:3]=np.eye(3);Hp[:,6:9]=-R@skew(ell)
    Hv=np.zeros((3,15));Hv[:,3:6]=np.eye(3);Hv[:,6:9]=-R@skew(u);Hv[:,12:15]=R@skew(ell)
    out=[('point_p',hp,Hp),('point_v',hv,Hv)]
    if axis_child is not None:
        E=tangent(axis_world);h=E.T@R@axis_child
        H=np.zeros((2,15));H[:,6:9]=-E.T@R@skew(axis_child)
        out.append(('axis',h,H))
    if guide is not None:
        c,d=guide;E=tangent(d)
        out += [('guide_p',E.T@(p+R@ell-c),E.T@Hp),('guide_v',E.T@(v+R@u),E.T@Hv)]
    if attitude_ref is not None:
        q=Rot.from_matrix(attitude_ref.T@R).as_quat()
        if q[3]<0:q=-q
        H=np.zeros((3,15));H[:,6:9]=q[3]*np.eye(3)+skew(q[:3])
        out.append(('fixed_attitude',2*q[:3],H))
    return out

def process_cov(R,dt,cfg):
    # Per-sample white-noise SD mapped to increments; RW SD per sqrt(second).
    G=np.zeros((15,12));G[:3,:3]=.5*R*dt**2;G[3:6,:3]=R*dt
    G[6:9,3:6]=-np.eye(3)*dt;G[9:12,6:9]=np.eye(3)*np.sqrt(dt);G[12:15,9:12]=np.eye(3)*np.sqrt(dt)
    s=np.repeat([cfg['acc_noise_sd'],cfg['gyro_noise_sd'],cfg['ba_rw_sd'],cfg['bg_rw_sd']],3)
    return (G*s**2)@G.T

def heading_null(p,v,R,support,g):
    """Unobservable error direction: rotation about gravity through the fixed support."""
    up=unit(-np.asarray(g))
    return np.r_[np.cross(up,p-support),np.cross(up,v),R.T@up,np.zeros(6)]

def reconstruct(body,acc,gyr,dt,cal,cfg,parent=None,parent_axis=None):
    N=len(acc);p=cal['p0'].copy();v=cal['v0'].copy();R=cal['R0'].copy()
    ba=np.zeros(3);bg=np.zeros(3);ell=cal['lever'];g=np.array(cfg['gravity'])
    sd=np.repeat(cfg['initial_state_sd'],3);P=np.diag(sd**2);I=np.eye(15)
    # Stage II: midpoint specific-force propagation and heading-consistent (OC-EKF) projection.
    fix=cfg.get('stage2_fix',{});midpoint=fix.get('propagation')=='midpoint'
    oc=bool(fix.get('oc_heading',False)) and parent is None and cfg['joint_types'][body-1]=='spherical'
    support=np.array(cfg.get('fixed_support',[0.,0.,0.]))
    n_prev=heading_null(p,v,R,support,g) if oc else None
    out={name:np.empty((N,3)) for name in ('p','v','ba','bg')};out['R']=np.empty((N,3,3))
    out['Pdiag']=np.empty((N,15));out['acc']=acc.copy();out['gyr']=gyr.copy()
    stats=dict(min_covariance_eigenvalue=1.,max_covariance_asymmetry=0.,max_correction_angle_rad=0.,max_rotation_orthogonality_error=0.)
    for k in range(N):
        if k:
            w=.5*(gyr[k-1]+gyr[k])-bg;f=.5*(acc[k-1]+acc[k])-ba
            Rf=R@Rot.from_rotvec(.5*w*dt).as_matrix() if midpoint else None
            R=R@Rot.from_rotvec(w*dt).as_matrix()
            if not midpoint:Rf=R
            aw=Rf@f+g;p=p+v*dt+.5*aw*dt**2;v=v+aw*dt
            B=Rf@skew(f);Phi=I.copy();Phi[:3,3:6]=np.eye(3)*dt
            Phi[:3,6:9]=-.5*B*dt**2;Phi[:3,9:12]=-.5*Rf*dt**2
            Phi[3:6,6:9]=-B*dt;Phi[3:6,9:12]=-Rf*dt
            Phi[6:9,6:9]=np.eye(3)-skew(w)*dt;Phi[6:9,12:15]=-np.eye(3)*dt
            if oc:
                # Phi' n_k = n_{k+1}, both null vectors at prior estimates (Hesch et al. 2014).
                n_new=heading_null(p,v,R,support,g)
                Phi=Phi-np.outer(Phi@n_prev-n_new,n_prev)/(n_prev@n_prev)
            P=Phi@P@Phi.T+process_cov(R,dt,cfg)
            if parent is None:jp=np.array(cfg.get('fixed_support',[0.,0.,0.]));jv=np.zeros(3)
            else:
                Rp=parent['R'][k];lp=cal['parent_lever'];wp=parent['gyr'][k]-parent['bg'][k]
                jp=parent['p'][k]+Rp@lp;jv=parent['v'][k]+Rp@np.cross(wp,lp)
            extra={}
            if 'axis' in cal and cfg.get('axis_updates',True):
                axworld=cal['R0']@cal['axis'] if body==1 else parent['R'][k]@parent_axis
                extra=dict(axis_child=cal['axis'],axis_world=axworld)
            if 'guide_direction' in cal:extra=dict(guide=(cal['guide_point'],cal['guide_direction']),attitude_ref=cal['R0'])
            blocks=residuals(p,v,R,ba,bg,gyr[k],ell,jp,jv,**extra)
            y=np.concatenate([b[1] for b in blocks]);H=np.vstack([b[2] for b in blocks])
            if oc:H=H-np.outer(H@n_new,n_new)/(n_new@n_new);n_prev=n_new
            V=block_diag(*[np.eye(len(b[1]))*cfg['constraint_sd'][b[0]]**2 for b in blocks])
            PH=P@H.T;S=H@PH+V;K=np.linalg.solve(S,PH.T).T
            dx=-K@y;A=I-K@H;P=A@P@A.T+K@V@K.T
            # Paper uses identity reset-Jacobian approximation; retained here.
            p+=dx[:3];v+=dx[3:6];R=R@Rot.from_rotvec(dx[6:9]).as_matrix();ba+=dx[9:12];bg+=dx[12:15]
            stats['max_correction_angle_rad']=max(stats['max_correction_angle_rad'],float(np.linalg.norm(dx[6:9])))
        for name,value in [('p',p),('v',v),('R',R),('ba',ba),('bg',bg)]:out[name][k]=value
        out['Pdiag'][k]=np.diag(P)
        if not all(np.isfinite(x).all() for x in (p,v,R,ba,bg,P)):raise FloatingPointError(f'Body {body}, sample {k}')
        if k%100==0 or k==N-1:
            stats['min_covariance_eigenvalue']=min(stats['min_covariance_eigenvalue'],float(np.linalg.eigvalsh((P+P.T)/2).min()))
            stats['max_covariance_asymmetry']=max(stats['max_covariance_asymmetry'],float(np.max(np.abs(P-P.T))))
            stats['max_rotation_orthogonality_error']=max(stats['max_rotation_orthogonality_error'],float(np.max(np.abs(R.T@R-np.eye(3)))))
    out['checks']=stats
    assert stats['min_covariance_eigenvalue']>-1e-10
    return out

def run_pipeline(measurements,cfg,on_stage=None):
    t=measurements['time'];dt=float(t[1]-t[0]);nc=int(round(cfg['calibration_s']/dt))
    calibrations={};histories={};timings={}
    for b in (1,2,3):
        print(f'Body {b}: Stage I',flush=True);start=time.perf_counter()
        parent=histories.get(b-1)
        cal=calibrate(b,measurements[f'acc{b}'][:nc],measurements[f'gyr{b}'][:nc],dt,cfg,
            parent=parent,known_child_lever=cfg['slider_lever'] if b==3 else None)
        t1=time.perf_counter()-start
        print(f"Body {b}: selected residual cost={cal['fit']['cost']:.6g}, converged={cal['fit']['converged']}",flush=True)
        print(f'Body {b}: Stage II',flush=True);start=time.perf_counter()
        hist=reconstruct(b,measurements[f'acc{b}'],measurements[f'gyr{b}'],dt,cal,cfg,
            parent=parent,parent_axis=calibrations[1]['axis'] if b==2 else None)
        calibrations[b]=cal;histories[b]=hist;timings[b]=dict(stageI_s=t1,stageII_s=time.perf_counter()-start)
        if on_stage:on_stage(b,cal,hist,timings[b])
    return calibrations,histories,timings
