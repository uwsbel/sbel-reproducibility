"""Stage I of the paper; accepts measurements and the already estimated parent only."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from core import skew, unit, tangent, increments, apply, fit_best, initial_guesses

STAGE1_KEEP=None  # Stage I gyroscope-range rule: mask over calibration samples, set by the runner
def range_keep(gyros,limit,half):
    w=np.max([np.linalg.norm(x,axis=1) for x in gyros],axis=0)
    return ~(np.convolve(w>limit,np.ones(2*half+1),'same')>0)

def inputs(acc,gyr,dt,window):
    w=savgol_filter(gyr,window,3,axis=0)
    alpha=np.gradient(w,dt,axis=0)
    A=np.array([skew(a)+skew(v)@skew(v) for a,v in zip(alpha,w)])
    return acc,w,A

def gauge(R0,g,local=None,target=None):
    # Choose a world azimuth; no evaluation reference enters this function.
    u=unit(-np.asarray(g));a=R0@(np.array([1.,0.,0.]) if local is None else local)
    b=np.array([1.,0.,0.]) if target is None else np.asarray(target)
    a=unit(a-u*np.dot(a,u));b=unit(b-u*np.dot(b,u))
    angle=np.arctan2(np.dot(u,np.cross(a,b)),np.dot(a,b))
    G=R.from_rotvec(u*angle).as_matrix()
    return G@R0,G


def _skew_v(v):
    z=np.zeros(len(v));return np.stack([np.stack([z,-v[:,2],v[:,1]],1),np.stack([v[:,2],z,-v[:,0]],1),np.stack([-v[:,1],v[:,0],z],1)],1)

def _A_of(gyr,dt,window):
    w=savgol_filter(gyr,window,3,axis=0);al=np.gradient(w,dt,axis=0);Sw=_skew_v(w)
    return _skew_v(al)+Sw@Sw

def eiv_Q(gyr,dt,window,keep,sd,seed,n):
    """Noise compensation: expected sum of E_k^T E_k for the regressor noise, by Monte Carlo through the processing."""
    from numpy.random import default_rng
    rng=default_rng(seed);A0=_A_of(gyr,dt,window);Q=np.zeros((3,3))
    for _ in range(n):
        E=_A_of(gyr+rng.normal(0,sd,gyr.shape),dt,window)-A0
        if keep is not None:E=E[keep]
        Q+=np.einsum('kji,kjl->il',E,E)
    return Q/n

def calibrate(body,acc,gyr,dt,cfg,parent=None,parent_axis=None):
    # Axial convention: estimated levers are projected perpendicular to the estimated axis.
    perp=lambda l,n:np.asarray(l)-(np.asarray(l)@np.asarray(n))*np.asarray(n)
    g=np.array(cfg['gravity']);nc=len(acc);rj=cfg['joint_types'][body-1]=='revolute'
    f,w,A=inputs(acc,gyr,dt,cfg['sg_windows'][body-1]);M=increments(gyr,dt)
    if parent is not None:
        Rp=parent['R'][:nc]
        fp,wp,Ap=inputs(parent['acc'][:nc]-parent['ba'][:nc],parent['gyr'][:nc]-parent['bg'][:nc],dt,cfg['sg_windows'][body-1])
    if rj:
        _,_,V=np.linalg.svd(w,full_matrices=False);nbase=V[0]
        if nbase[np.argmax(np.abs(nbase))]<0:nbase=-nbase
        T=tangent(nbase)
        def axis(x):return R.from_rotvec(T@x).apply(nbase)
    def fun(x):
        Rc=R.from_rotvec(x[:3]).as_matrix()@M
        if parent is None:
            ra=apply(Rc,f+np.einsum('nij,j->ni',A,x[3:6]))+g;wr=w
        else:
            ra=apply(Rc,f+np.einsum('nij,j->ni',A,x[6:9]))-apply(Rp,fp+np.einsum('nij,j->ni',Ap,x[3:6]))
            wr=w-apply(np.transpose(Rc,(0,2,1)),apply(Rp,wp)) if rj else None
        if STAGE1_KEEP is not None:ra=ra[STAGE1_KEEP]
        if not rj:return ra.ravel()
        n=axis(x[-2:]);rw=wr-(wr@n)[:,None]*n
        if STAGE1_KEEP is not None:rw=rw[STAGE1_KEEP]
        return np.r_[ra.ravel(),cfg['axis_residual_multipliers'][body-1]*rw.ravel()]
    rng=np.random.default_rng(cfg['initialization_seed']+body)
    if cfg.get('random_starts',False):
        starts=[np.r_[R.random(random_state=rng).as_rotvec(),rng.uniform(-.5,.5,3 if parent is None else 6),np.zeros(2) if rj else []] for _ in range(cfg['n_starts'])]
    else:
        bases=initial_guesses(acc,g)
        bases += [(R.from_rotvec(unit(g)*a)*R.from_rotvec(bases[1])).as_rotvec() for a in [np.pi/2,np.pi,3*np.pi/2]]
        starts=[np.r_[r,rng.uniform(-.1,.1,3 if parent is None else 6),np.zeros(2) if rj else []] for r in bases]
    x,info=fit_best(fun,starts,cfg['max_nfev'][body-1],cfg['solver_tolerance'])
    _sd=cfg.get('stage1_eiv_gyro_sd',cfg['gyro_noise_sd'])
    if cfg.get('stage1_eiv',False) and _sd>0:
        from scipy.optimize import minimize
        seed=cfg['stage1_eiv_seed']+10*body;win=cfg['sg_windows'][body-1];nmc=cfg['stage1_eiv_mc']
        Qc=eiv_Q(gyr,dt,win,STAGE1_KEEP,_sd,seed,nmc)
        Qp=eiv_Q(parent['gyr'][:nc]-parent['bg'][:nc],dt,win,STAGE1_KEEP,_sd,seed+1,nmc) if parent is not None else None
        if cfg.get('stage1_eiv_project_axis',False) and rj:
            _P=lambda nn:np.eye(3)-np.outer(nn,nn)
            _nc=axis(x[-2:]);Qc=_P(_nc)@Qc@_P(_nc)
            if parent is not None and parent_axis is not None:_np=np.asarray(parent_axis);Qp=_P(_np)@Qp@_P(_np)
        def Jc(z):
            r=fun(z);return .5*(r@r-(z[6:9]@Qc@z[6:9]+z[3:6]@Qp@z[3:6] if parent is not None else z[3:6]@Qc@z[3:6]))
        sb=minimize(Jc,x,method='BFGS',options=dict(gtol=cfg['stage1_eiv_gtol'],maxiter=2000))
        info['eiv']=dict(x_uncompensated=x.tolist(),Jc=float(sb.fun),message=str(sb.message),nit=int(sb.nit),sd=_sd,Qc=Qc.tolist(),Qp=None if Qp is None else Qp.tolist())
        x=sb.x
    R0=R.from_rotvec(x[:3]).as_matrix();G=np.eye(3)
    if parent is None:
        R0,G=gauge(R0,g,axis(x[-2:]) if rj else None,cfg.get('heading_axis_target') if rj else None)
        ell=perp(x[3:6],axis(x[-2:])) if rj else x[3:6];p0=np.array(cfg['fixed_support'])-R0@ell;v0=-R0@np.cross(gyr[0],ell)
        result=dict(R0=R0,lever=ell,lever_raw=x[3:6],p0=p0,v0=v0,gauge_rotation=G,fit=info)
    else:
        lp=perp(x[3:6],parent_axis) if rj else x[3:6];ell=perp(x[6:9],axis(x[-2:])) if rj else x[6:9]
        p0=parent['p'][0]+Rp[0]@lp-R0@ell
        v0=parent['v'][0]+Rp[0]@np.cross(parent['gyr'][0]-parent['bg'][0],lp)-R0@np.cross(gyr[0],ell)
        result=dict(R0=R0,lever=ell,parent_lever=lp,lever_raw=x[6:9],parent_lever_raw=x[3:6],p0=p0,v0=v0,fit=info)
    if rj:result['axis']=axis(x[-2:])
    return result
