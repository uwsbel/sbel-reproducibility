"""Numba translation of the spherical-joint 15-state Stage II equations.

Timing excludes archive I/O and periodic numerical diagnostics. Numerical
equivalence to core.reconstruct is checked before any timings are reported.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def skew3(v):
    a=np.zeros((3,3));a[0,1]=-v[2];a[0,2]=v[1];a[1,0]=v[2];a[1,2]=-v[0];a[2,0]=-v[1];a[2,1]=v[0]
    return a

@njit(cache=True)
def exp3(v):
    s=skew3(v);a=np.sqrt(np.dot(v,v))
    if a<1e-7:return np.eye(3)+(1-a*a/6)*s+(.5-a*a/24)*(s@s)
    return np.eye(3)+np.sin(a)/a*s+(1-np.cos(a))/(a*a)*(s@s)

@njit(cache=True)
def filter_spherical(acc,gyr,dt,p0,v0,R0,ell,gravity,sd,noise,sigma_p,sigma_v,jp,jv,midpoint=False,oc=False,support=np.zeros(3)):
    n=len(acc);p=p0.copy();v=v0.copy();r=R0.copy();ba=np.zeros(3);bg=np.zeros(3)
    P=np.diag(sd**2);eye=np.eye(15);e3=np.eye(3)
    op=np.empty((n,3));ov=np.empty((n,3));orr=np.empty((n,3,3));oba=np.empty((n,3));obg=np.empty((n,3))
    V=np.diag(np.array([sigma_p**2]*3+[sigma_v**2]*3))
    # Stage II: midpoint specific force; heading-consistent (OC-EKF) projection for the spherical anchor.
    up=-gravity/np.sqrt(np.dot(gravity,gravity));z6=np.zeros(6)
    n_prev=np.concatenate((np.cross(up,p-support),np.cross(up,v),r.T@up,z6))
    for k in range(n):
        if k:
            w=.5*(gyr[k-1]+gyr[k])-bg;f=.5*(acc[k-1]+acc[k])-ba
            rf=r@exp3(.5*w*dt) if midpoint else r
            r=r@exp3(w*dt)
            if not midpoint:rf=r
            aw=rf@f+gravity;p=p+v*dt+.5*aw*dt**2;v=v+aw*dt
            B=rf@skew3(f);phi=eye.copy();phi[:3,3:6]=e3*dt
            phi[:3,6:9]=-.5*B*dt**2;phi[:3,9:12]=-.5*rf*dt**2
            phi[3:6,6:9]=-B*dt;phi[3:6,9:12]=-rf*dt
            phi[6:9,6:9]=e3-skew3(w)*dt;phi[6:9,12:15]=-e3*dt
            if oc:
                n_new=np.concatenate((np.cross(up,p-support),np.cross(up,v),r.T@up,z6))
                phi=phi-np.outer(phi@n_prev-n_new,n_prev)/np.dot(n_prev,n_prev)
            G=np.zeros((15,12));G[:3,:3]=.5*r*dt**2;G[3:6,:3]=r*dt
            G[6:9,3:6]=-e3*dt;G[9:12,6:9]=e3*np.sqrt(dt);G[12:15,9:12]=e3*np.sqrt(dt)
            P=phi@P@phi.T+(G*noise**2)@G.T
            u=np.cross(gyr[k]-bg,ell);hp=p+r@ell-jp[k];hv=v+r@u-jv[k]
            y=np.concatenate((hp,hv));H=np.zeros((6,15));H[:3,:3]=e3;H[:3,6:9]=-r@skew3(ell)
            H[3:6,3:6]=e3;H[3:6,6:9]=-r@skew3(u);H[3:6,12:15]=r@skew3(ell)
            if oc:
                H=H-np.outer(H@n_new,n_new)/np.dot(n_new,n_new);n_prev=n_new
            PH=P@H.T;S=H@PH+V;K=np.linalg.solve(S,PH.T).T;dx=-K@y;A=eye-K@H;P=A@P@A.T+K@V@K.T
            p+=dx[:3];v+=dx[3:6];r=r@exp3(dx[6:9]);ba+=dx[9:12];bg+=dx[12:15]
        op[k]=p;ov[k]=v;orr[k]=r;oba[k]=ba;obg[k]=bg
    return op,ov,orr,oba,obg

def run_chain(m,cals,cfg):
    dt=float(np.median(np.diff(m['time'])));n=len(m['time']);hist={}
    sd=np.repeat(cfg['initial_state_sd'],3);noise=np.repeat([cfg['acc_noise_sd'],cfg['gyro_noise_sd'],cfg['ba_rw_sd'],cfg['bg_rw_sd']],3)
    fix=cfg.get('stage2_fix',{});midpoint=fix.get('propagation')=='midpoint'
    for b in range(1,cfg['n_bodies']+1):
        c=cals[b]
        oc=bool(fix.get('oc_heading',False)) and b==1 and cfg['joint_types'][0]=='spherical'
        if b==1:jp=np.tile(cfg['fixed_support'],(n,1));jv=np.zeros((n,3))
        else:
            h=hist[b-1];lp=c['parent_lever'];rp=h['R']
            jp=h['p']+np.einsum('nij,j->ni',rp,lp)
            jv=h['v']+np.einsum('nij,nj->ni',rp,np.cross(m[f'gyr{b-1}']-h['bg'],lp))
        values=filter_spherical(m[f'acc{b}'],m[f'gyr{b}'],dt,c['p0'],c['v0'],c['R0'],c['lever'],np.asarray(cfg['gravity'],float),sd,noise,cfg['constraint_sd']['point_p'],cfg['constraint_sd']['point_v'],jp,jv,midpoint,oc,np.asarray(cfg['fixed_support'],float))
        hist[b]=dict(zip(['p','v','R','ba','bg'],values))
    return hist
