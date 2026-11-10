# Independent JT implementation for the three-link benchmark.
# Numerics: midpoint inputs, exact attitude transition,
# SO(3) reset Jacobian, finite checks, sampled covariance diagnostics.
# Discrete process noise Qd and measurement covariance are fixed values, not tuned.
from __future__ import annotations

# ========== Imports ==========
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter 
from scipy.linalg import lstsq
from scipy.optimize import least_squares 
from scipy.linalg import block_diag
import time
from numba import njit
# ==============================================================================
# Numba JIT math helpers
# ==============================================================================
@njit(cache=True, fastmath=False)
def _skew_numba(v, out):
    out[0, 0] = 0.0;     out[0, 1] = -v[2];   out[0, 2] = v[1]
    out[1, 0] = v[2];    out[1, 1] = 0.0;     out[1, 2] = -v[0]
    out[2, 0] = -v[1];   out[2, 1] = v[0];    out[2, 2] = 0.0

@njit(cache=True, fastmath=False)
def quat_mult_numba(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

@njit(cache=True, fastmath=False)
def quat_to_rotmat_numba(q, out):
    w, x, y, z = q[0], q[1], q[2], q[3]
    x2, y2, z2 = x*2, y*2, z*2
    xx, yy, zz = x*x2, y*y2, z*z2
    xy, xz, yz = x*y2, x*z2, y*z2
    wx, wy, wz = w*x2, w*y2, w*z2
    out[0, 0] = 1.0 - (yy + zz); out[0, 1] = xy - wz;       out[0, 2] = xz + wy
    out[1, 0] = xy + wz;       out[1, 1] = 1.0 - (xx + zz); out[1, 2] = yz - wx
    out[2, 0] = xz - wy;       out[2, 1] = yz + wx;       out[2, 2] = 1.0 - (xx + yy)

@njit(cache=True, fastmath=False)
def _exp_quat_numba(dth):
    angle = np.sqrt(dth[0]*dth[0] + dth[1]*dth[1] + dth[2]*dth[2])
    if angle < 1e-8: return np.array([1.0, 0.0, 0.0, 0.0])
    s = np.sin(angle * 0.5) / angle
    return np.array([np.cos(angle * 0.5), dth[0]*s, dth[1]*s, dth[2]*s])

def mix3d(true_vec, norm_val, ratio, directions):
    return true_vec + (norm_val * ratio * np.array(directions))

@njit(cache=True)
def rot_exp(v):
    s=np.zeros((3,3)); _skew_numba(v,s); a=np.sqrt(np.dot(v,v))
    if a < 1e-7: return np.eye(3)+(1-a*a/6)*s+(.5-a*a/24)*(s@s)
    return np.eye(3)+np.sin(a)/a*s+(1-np.cos(a))/(a*a)*(s@s)

@njit(cache=True)
def right_jacobian(v):
    s=np.zeros((3,3)); _skew_numba(v,s); a=np.sqrt(np.dot(v,v))
    if a < 1e-5: return np.eye(3)-(.5-a*a/24)*s+(1/6-a*a/120)*(s@s)
    return np.eye(3)-(1-np.cos(a))/(a*a)*s+(a-np.sin(a))/(a*a*a)*(s@s)

@njit(cache=True)
def propagate_body(r,v,q,acc_left,gyr_left,acc_right,gyr_right,dt,g,stable):
    rot=np.zeros((3,3)); quat_to_rotmat_numba(q,rot)
    if stable:
        w=.5*(gyr_left+gyr_right); f=.5*(acc_left+acc_right)
        rotated_f=rot_exp(w*dt*.5)@f
        aw=rot@rotated_f+g
        skew=np.zeros((3,3)); _skew_numba(rotated_f,skew)
        coupling=-rot@skew
        attitude_transition=rot_exp(-w*dt)
    else:
        w=gyr_left; aw=rot@acc_left+g
        skew=np.zeros((3,3)); _skew_numba(acc_left,skew); coupling=-rot@skew
        _skew_numba(w,skew); attitude_transition=np.eye(3)-skew*dt
    phi=np.eye(9); phi[:3,3:6]=np.eye(3)*dt
    phi[:3,6:9]=.5*coupling*dt*dt; phi[3:6,6:9]=coupling*dt; phi[6:9,6:9]=attitude_transition
    qnew=quat_mult_numba(q,_exp_quat_numba(w*dt)); qnew=qnew/np.sqrt(np.sum(qnew*qnew))
    return r+v*dt+.5*aw*dt*dt,v+aw*dt,qnew,phi

@njit(cache=True, fastmath=False)
def constraint_terms(r1,v1,q1,r2,v2,q2,r3,v3,q3,L1p,L1d,L2p,L2d,L3p,gyr1,gyr2,gyr3,pos_fixed_SJ,fix_jacobian=True):
    Rm1=np.zeros((3,3)); Rm2=np.zeros((3,3)); Rm3=np.zeros((3,3))
    quat_to_rotmat_numba(q1,Rm1); quat_to_rotmat_numba(q2,Rm2); quat_to_rotmat_numba(q3,Rm3)
    I3=np.eye(3); H=np.zeros((18,42)); y=np.zeros(18); skew_temp=np.zeros((3,3))
    # --- Residuals y ---
    w1_x_L1p = np.cross(gyr1, L1p); w1_x_L1d = np.cross(gyr1, L1d)
    w2_x_L2p = np.cross(gyr2, L2p); w2_x_L2d = np.cross(gyr2, L2d)
    w3_x_L3p = np.cross(gyr3, L3p)

    y[0:3] = (r1 + Rm1 @ L1p) - pos_fixed_SJ
    y[3:6] = (v1 + Rm1 @ w1_x_L1p)
    
    y[6:9] = (r2 + Rm2 @ L2p) - (r1 + Rm1 @ L1d)
    y[9:12] = (v2 + Rm2 @ w2_x_L2p) - (v1 + Rm1 @ w1_x_L1d)
    
    y[12:15] = (r3 + Rm3 @ L3p) - (r2 + Rm2 @ L2d)
    y[15:18] = (v3 + Rm3 @ w3_x_L3p) - (v2 + Rm2 @ w2_x_L2d)

    # --- Jacobian H ---
    H[:, :] = 0.0 # Clear
    
    # SJ 1 (Fixed)
    H[0:3, 0:3] = I3
    _skew_numba(L1p, skew_temp); H[0:3, 6:9] = -Rm1 @ skew_temp
    H[0:3, 27:30] = Rm1

    H[3:6, 3:6] = I3
    _skew_numba(w1_x_L1p, skew_temp); H[3:6, 6:9] = -Rm1 @ skew_temp
    if fix_jacobian:
        _skew_numba(gyr1, skew_temp); H[3:6, 27:30] = Rm1 @ skew_temp

    # SJ 2 (Body 1 -> Body 2)
    H[6:9, 0:3] = -I3
    _skew_numba(L1d, skew_temp); H[6:9, 6:9] = Rm1 @ skew_temp
    H[6:9, 9:12] = I3
    _skew_numba(L2p, skew_temp); H[6:9, 15:18] = -Rm2 @ skew_temp
    H[6:9, 30:33] = -Rm1; H[6:9, 33:36] = Rm2

    H[9:12, 3:6] = -I3
    _skew_numba(w1_x_L1d, skew_temp); H[9:12, 6:9] = Rm1 @ skew_temp
    H[9:12, 12:15] = I3
    _skew_numba(w2_x_L2p, skew_temp); H[9:12, 15:18] = -Rm2 @ skew_temp

    if fix_jacobian:
        _skew_numba(gyr1, skew_temp); H[9:12, 30:33] = -Rm1 @ skew_temp
        _skew_numba(gyr2, skew_temp); H[9:12, 33:36] = Rm2 @ skew_temp

    # SJ 3 (Body 2 -> Body 3)
    H[12:15, 9:12] = -I3
    _skew_numba(L2d, skew_temp); H[12:15, 15:18] = Rm2 @ skew_temp
    H[12:15, 18:21] = I3
    _skew_numba(L3p, skew_temp); H[12:15, 24:27] = -Rm3 @ skew_temp
    H[12:15, 36:39] = -Rm2; H[12:15, 39:42] = Rm3

    H[15:18, 12:15] = -I3
    _skew_numba(w2_x_L2d, skew_temp); H[15:18, 15:18] = Rm2 @ skew_temp
    H[15:18, 21:24] = I3
    _skew_numba(w3_x_L3p, skew_temp); H[15:18, 24:27] = -Rm3 @ skew_temp

    if fix_jacobian:
        _skew_numba(gyr2, skew_temp); H[15:18, 36:39] = -Rm2 @ skew_temp
        _skew_numba(gyr3, skew_temp); H[15:18, 39:42] = Rm3 @ skew_temp
    return y,H

@njit(cache=True)
def heading_null_jt(r1,v1,q1,r2,v2,q2,r3,v3,q3,pos_fixed_SJ,g_vec):
    # JT+OC: common heading rotation of the whole chain about the vertical through the support.
    u=-g_vec/np.sqrt(np.sum(g_vec*g_vec)); z=np.zeros(42); Rm=np.zeros((3,3))
    rs=(r1,r2,r3); vs=(v1,v2,v3); qs=(q1,q2,q3)
    for i in range(3):
        quat_to_rotmat_numba(qs[i],Rm); o=9*i
        z[o:o+3]=np.cross(u,rs[i]-pos_fixed_SJ); z[o+3:o+6]=np.cross(u,vs[i]); z[o+6:o+9]=Rm.T@u
    return z

# ==============================================================================
# Numba JIT: 42-state IEKF for JointTracker (compiled)
# ==============================================================================
@njit(cache=True, fastmath=False)
def run_jointtracker_3link_numba_core(
    acc1, gyr1, acc2, gyr2, acc3, gyr3, dt, g_vec, pos_fixed_SJ,
    r1, v1, q1, r2, v2, q2, r3, v3, q3,
    L1p, L1d, L2p, L2d, L3p,
    P, Qd, R_meas, n_iters, fix_jacobian, aligned_time, stable_propagation, exact_reset, collect_diagnostics, oc
):
    N = len(acc1)
    
    # state histories
    hist_r1 = np.zeros((N, 3)); hist_v1 = np.zeros((N, 3)); hist_q1 = np.zeros((N, 4))
    hist_r2 = np.zeros((N, 3)); hist_v2 = np.zeros((N, 3)); hist_q2 = np.zeros((N, 4))
    hist_r3 = np.zeros((N, 3)); hist_v3 = np.zeros((N, 3)); hist_q3 = np.zeros((N, 4))
    hist_L1p = np.zeros((N, 3)); hist_L1d = np.zeros((N, 3)); 
    hist_L2p = np.zeros((N, 3)); hist_L2d = np.zeros((N, 3)); hist_L3p = np.zeros((N, 3))

    diagnostic=np.zeros((N//100+1,9)); diagnostic_count=0
    I3 = np.eye(3)
    I42 = np.eye(42)
    Phi = np.eye(42)
    H = np.zeros((18, 42))
    y = np.zeros(18)
    
    # Work arrays
    Rm1 = np.zeros((3, 3)); Rm2 = np.zeros((3, 3)); Rm3 = np.zeros((3, 3))
    skew_temp = np.zeros((3, 3))
    z_prev = heading_null_jt(r1,v1,q1,r2,v2,q2,r3,v3,q3,pos_fixed_SJ,g_vec)
    z_new = z_prev.copy()

    for k in range(N):
        if aligned_time and k == 0:
            hist_r1[k]=r1; hist_v1[k]=v1; hist_q1[k]=q1
            hist_r2[k]=r2; hist_v2[k]=v2; hist_q2[k]=q2
            hist_r3[k]=r3; hist_v3[k]=v3; hist_q3[k]=q3
            hist_L1p[k]=L1p; hist_L1d[k]=L1d; hist_L2p[k]=L2p; hist_L2d[k]=L2d; hist_L3p[k]=L3p
            continue
        prop_index = k-1 if aligned_time else k
        # ================= A. Propagation =================
        Phi[:, :] = I42 # reset to identity
        
        r1,v1,q1,phi1=propagate_body(r1,v1,q1,acc1[prop_index],gyr1[prop_index],acc1[k],gyr1[k],dt,g_vec,stable_propagation)
        r2,v2,q2,phi2=propagate_body(r2,v2,q2,acc2[prop_index],gyr2[prop_index],acc2[k],gyr2[k],dt,g_vec,stable_propagation)
        r3,v3,q3,phi3=propagate_body(r3,v3,q3,acc3[prop_index],gyr3[prop_index],acc3[k],gyr3[k],dt,g_vec,stable_propagation)
        Phi[:9,:9]=phi1; Phi[9:18,9:18]=phi2; Phi[18:27,18:27]=phi3
        if oc:
            z_new = heading_null_jt(r1,v1,q1,r2,v2,q2,r3,v3,q3,pos_fixed_SJ,g_vec)
            Phi = Phi - np.outer(Phi @ z_prev - z_new, z_prev) / (z_prev @ z_prev)

        P = Phi @ P @ Phi.T + Qd

        # ================= B. IEKF Update =================
        for _ in range(n_iters):
            quat_to_rotmat_numba(q1, Rm1)
            quat_to_rotmat_numba(q2, Rm2)
            quat_to_rotmat_numba(q3, Rm3)

            y,H = constraint_terms(r1,v1,q1,r2,v2,q2,r3,v3,q3,L1p,L1d,L2p,L2d,L3p,
                                   gyr1[k],gyr2[k],gyr3[k],pos_fixed_SJ,fix_jacobian)
            if oc:
                H = H - np.outer(H @ z_new, z_new) / (z_new @ z_new)

            # Update Math
            PHt = P @ H.T
            S = H @ PHt + R_meas
            K = np.linalg.solve(S.T, PHt.T).T 
            dx = K @ (-y)

            # Inject
            r1 += dx[0:3]; v1 += dx[3:6]
            q1 = quat_mult_numba(q1, _exp_quat_numba(dx[6:9]))
            q1 /= np.sqrt(np.sum(q1**2))
            
            r2 += dx[9:12]; v2 += dx[12:15]
            q2 = quat_mult_numba(q2, _exp_quat_numba(dx[15:18]))
            q2 /= np.sqrt(np.sum(q2**2))
            
            r3 += dx[18:21]; v3 += dx[21:24]
            q3 = quat_mult_numba(q3, _exp_quat_numba(dx[24:27]))
            q3 /= np.sqrt(np.sum(q3**2))
            
            L1p += dx[27:30]; L1d += dx[30:33]
            L2p += dx[33:36]; L2d += dx[36:39]
            L3p += dx[39:42]

        I_KH = I42 - K @ H
        P = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
        if exact_reset:
            reset=I42.copy(); reset[6:9,6:9]=right_jacobian(dx[6:9]); reset[15:18,15:18]=right_jacobian(dx[15:18]); reset[24:27,24:27]=right_jacobian(dx[24:27])
            P=reset@P@reset.T
        if collect_diagnostics and (k%100 == 0 or k==N-1):
            if not np.isfinite(P).all() or not np.isfinite(dx).all(): raise FloatingPointError('Non-finite JT state/covariance update')
            eigen_p=np.linalg.eigvalsh(.5*(P+P.T)); eigen_s=np.linalg.eigvalsh(.5*(S+S.T))
            diagnostic[diagnostic_count,0]=k*dt
            diagnostic[diagnostic_count,1]=eigen_p[0]
            diagnostic[diagnostic_count,2]=np.max(np.abs(P-P.T))
            diagnostic[diagnostic_count,3]=eigen_s[-1]/eigen_s[0]
            diagnostic[diagnostic_count,4]=max(np.linalg.norm(dx[6:9]),np.linalg.norm(dx[15:18]),np.linalg.norm(dx[24:27]))
            diagnostic[diagnostic_count,5]=max(np.linalg.norm(L1p),np.linalg.norm(L1d),np.linalg.norm(L2p),np.linalg.norm(L2d),np.linalg.norm(L3p))
            yp,_=constraint_terms(r1,v1,q1,r2,v2,q2,r3,v3,q3,L1p,L1d,L2p,L2d,L3p,gyr1[k],gyr2[k],gyr3[k],pos_fixed_SJ,True)
            diagnostic[diagnostic_count,6]=max(np.linalg.norm(yp[:3]),np.linalg.norm(yp[6:9]),np.linalg.norm(yp[12:15]))
            diagnostic[diagnostic_count,7]=max(np.linalg.norm(yp[3:6]),np.linalg.norm(yp[9:12]),np.linalg.norm(yp[15:18]))
            diagnostic[diagnostic_count,8]=min(eigen_s[0],1.)
            diagnostic_count+=1
        P = 0.5 * (P + P.T)
        if oc:
            z_prev = z_new
        
        # Log
        hist_r1[k] = r1; hist_v1[k] = v1; hist_q1[k] = q1
        hist_r2[k] = r2; hist_v2[k] = v2; hist_q2[k] = q2
        hist_r3[k] = r3; hist_v3[k] = v3; hist_q3[k] = q3
        hist_L1p[k] = L1p; hist_L1d[k] = L1d
        hist_L2p[k] = L2p; hist_L2d[k] = L2d; hist_L3p[k] = L3p

    return hist_r1, hist_v1, hist_q1, hist_r2, hist_v2, hist_q2, hist_r3, hist_v3, hist_q3, hist_L1p, hist_L1d, hist_L2p, hist_L2d, hist_L3p, diagnostic[:diagnostic_count]


# Python wrapper (called by the run scripts)
def run_jointtracker_3link_numba_wrapper(acc1, gyr1, acc2, gyr2, acc3, gyr3, dt, g_vec, pos_fixed_SJ, init_states, init_params, R_diag, Qd_diag, n_iters=1, fix_jacobian=True, aligned_time=True, stable_propagation=True, exact_reset=True, collect_diagnostics=True, oc=False, Qd_kin=None):
    if n_iters != 1:
        raise ValueError('This minimal repair supports one EKF update only; repeated updates are not a valid IEKF iteration.')
    arrays=[acc1,gyr1,acc2,gyr2,acc3,gyr3]
    if not np.isfinite(dt) or dt <= 0 or any(x.shape != arrays[0].shape or x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected matching finite (N,3) IMU arrays and a positive dt.')
    # Covariance matrices (built in Python, passed to Numba)
    P = np.eye(42) * 1e-3
    Q_b = np.diag([Qd_diag[0]]*3 + [Qd_diag[1]]*3 + [Qd_diag[2]]*3)
    if Qd_kin is not None:
        Q_b = np.array(Qd_kin, dtype=float)  # JT-Qn: sensor-noise process noise, STEP discretization
    Q_p = np.diag([Qd_diag[3]]*3)
    
    # assembled by hand instead of block_diag
    Qd = np.zeros((42, 42))
    Qd[0:9, 0:9] = Q_b; Qd[9:18, 9:18] = Q_b; Qd[18:27, 18:27] = Q_b
    Qd[27:30, 27:30] = Q_p; Qd[30:33, 30:33] = Q_p; Qd[33:36, 33:36] = Q_p
    Qd[36:39, 36:39] = Q_p; Qd[39:42, 39:42] = Q_p
    
    R_meas = np.eye(18) * (R_diag**2)

    # contiguous arrays for Numba
    res = run_jointtracker_3link_numba_core(
        np.ascontiguousarray(acc1), np.ascontiguousarray(gyr1),
        np.ascontiguousarray(acc2), np.ascontiguousarray(gyr2),
        np.ascontiguousarray(acc3), np.ascontiguousarray(gyr3),
        dt, g_vec, pos_fixed_SJ,
        init_states['r1'].copy(), init_states['v1'].copy(), init_states['q1'].copy(),
        init_states['r2'].copy(), init_states['v2'].copy(), init_states['q2'].copy(),
        init_states['r3'].copy(), init_states['v3'].copy(), init_states['q3'].copy(),
        init_params['L1p'].copy(), init_params['L1d'].copy(),
        init_params['L2p'].copy(), init_params['L2d'].copy(),
        init_params['L3p'].copy(),
        P, Qd, R_meas, n_iters, fix_jacobian, aligned_time, stable_propagation, exact_reset, collect_diagnostics, oc
    )
    
    hist = {
        'r1': res[0], 'v1': res[1], 'q1': res[2],
        'r2': res[3], 'v2': res[4], 'q2': res[5],
        'r3': res[6], 'v3': res[7], 'q3': res[8],
        'L1p': res[9], 'L1d': res[10], 'L2p': res[11], 'L2d': res[12], 'L3p': res[13], 'numerical_checks':res[14]
    }
    return hist


