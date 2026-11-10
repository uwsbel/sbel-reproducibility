"""Structure-exploiting N-body coupled filters, used for accuracy runs and timing.
Same equations as jt_nbody_dense.py. What is exploited:
- Phi: each body's 15-state block is applied to the rows and columns of P; lever states are constant.
- H: a joint involves two bodies and two levers, so PH^T and HPH^T use only those rows (fixed sparsity pattern).
- Joseph form in expanded algebra, P - KC - (KC)^T + KSK^T, and the attitude reset only on attitude rows and columns.
- Heading-consistent projection (JT-BHS): rank-one corrections of Phi and H.
P stays dense; the K, C products are dense. The solve is LU, as in STEP. joseph=False gives P - KC, for a timing sensitivity."""
import numpy as np
from numba import njit
import nbody_common as nc
from jt_checked import _skew_numba, quat_mult_numba, quat_to_rotmat_numba, _exp_quat_numba, right_jacobian
from jt_bias import prop_bias
from jt_nbody_dense import heading_dir


def h_pattern(N, est):
    """Columns of the nonzero entries of each row of H, in the order fill_H writes them."""
    nm, nl, n = nc.layout(N, est); lo = nm if est else -1; bo = nm + nl
    rows = [[] for _ in range(6 * N)]
    for j in range(N):
        row = 6 * j; rj = 9 * j; g = bo + 6 * j + 3; rp = 9 * (j - 1); gp = bo + 6 * (j - 1) + 3
        for a in range(3):
            rows[row + a] += [rj + a] + [rj + 6 + c for c in range(3)]
            rows[row + 3 + a] += [rj + 3 + a] + [rj + 6 + c for c in range(3)]
            if est:
                c0 = lo + 6 * j
                rows[row + a] += [c0 + c for c in range(3)]; rows[row + 3 + a] += [c0 + c for c in range(3)]
            rows[row + 3 + a] += [g + c for c in range(3)]
            if j > 0:
                rows[row + a] += [rp + a] + [rp + 6 + c for c in range(3)]
                rows[row + 3 + a] += [rp + 3 + a] + [rp + 6 + c for c in range(3)]
                if est:
                    c1 = lo + 3 * (2 * j - 1)
                    rows[row + a] += [c1 + c for c in range(3)]; rows[row + 3 + a] += [c1 + c for c in range(3)]
                rows[row + 3 + a] += [gp + c for c in range(3)]
    nnz = np.array([len(x) for x in rows], dtype=np.int64); cols = np.zeros((6 * N, int(nnz.max())), dtype=np.int64)
    for i, x in enumerate(rows): cols[i, :len(x)] = x
    return cols, nnz


@njit(cache=False)
def _axpy(alpha, x, y):
    for i in range(y.shape[0]):
        y[i] += alpha * x[i]


@njit(cache=False)
def _setb(vals, row, pos, A, sign):
    for a in range(3):
        for c in range(3):
            vals[row + a, pos + c] = sign * A[a, c]


@njit(cache=False)
def fill_H(N, est, r, v, q, L, w, pivot, y, vals, Rs, s):
    """Residuals into y and the nonzero values of H into vals, in the order of h_pattern."""
    for i in range(N): quat_to_rotmat_numba(q[i], Rs[i])
    for j in range(N):
        row = 6 * j; Rj = Rs[j]; Lp = L[2 * j]; wxL = np.cross(w[j], Lp)
        y[row:row + 3] = r[j] + Rj @ Lp; y[row + 3:row + 6] = v[j] + Rj @ wxL
        _skew_numba(Lp, s); A1 = Rj @ s
        _skew_numba(wxL, s); A3 = Rj @ s
        p = 0; pv = 0
        for a in range(3): vals[row + a, 0] = 1.0; vals[row + 3 + a, 0] = 1.0
        _setb(vals, row, 1, A1, -1.0); _setb(vals, row + 3, 1, A3, -1.0); p = 4; pv = 4
        if est:
            _setb(vals, row, p, Rj, 1.0); p += 3
            _skew_numba(w[j], s); B = Rj @ s; _setb(vals, row + 3, pv, B, 1.0); pv += 3
        _setb(vals, row + 3, pv, A1, 1.0); pv += 3
        if j == 0:
            y[row:row + 3] -= pivot
        else:
            Rp = Rs[j - 1]; Ld = L[2 * j - 1]; wxLd = np.cross(w[j - 1], Ld)
            y[row:row + 3] -= r[j - 1] + Rp @ Ld; y[row + 3:row + 6] -= v[j - 1] + Rp @ wxLd
            _skew_numba(Ld, s); A2 = Rp @ s
            _skew_numba(wxLd, s); A4 = Rp @ s
            for a in range(3): vals[row + a, p] = -1.0; vals[row + 3 + a, pv] = -1.0
            p += 1; pv += 1
            _setb(vals, row, p, A2, 1.0); p += 3
            _setb(vals, row + 3, pv, A4, 1.0); pv += 3
            if est:
                _setb(vals, row, p, Rp, -1.0); p += 3
                _skew_numba(w[j - 1], s); B = Rp @ s; _setb(vals, row + 3, pv, B, -1.0); pv += 3
            _setb(vals, row + 3, pv, A2, -1.0); pv += 3


@njit(cache=False)
def apply_H(cols, nnz, vals, X, out):
    """out = H X, one sparse row at a time; rows of X are contiguous."""
    for row in range(cols.shape[0]):
        o = out[row]; o[:] = 0.0
        for idx in range(nnz[row]):
            _axpy(vals[row, idx], X[cols[row, idx]], o)


@njit(cache=False)
def phi_rows(P, o, bi, dt, f, a, g, scratch):
    """Rows of one body of P <- Phi_body P. f: 9 x 9 motion block; a, g: 9 x 3 bias blocks (structure of prop_bias)."""
    for i in range(3):
        row = P[o + i]
        _axpy(dt, P[o + 3 + i], row)
        for c in range(3):
            _axpy(f[i, 6 + c], P[o + 6 + c], row); _axpy(a[i, c], P[bi + c], row)
    for i in range(3):
        row = P[o + 3 + i]
        for c in range(3):
            _axpy(f[3 + i, 6 + c], P[o + 6 + c], row); _axpy(a[3 + i, c], P[bi + c], row)
    for i in range(3):
        row = scratch[i]; row[:] = 0.0
        for c in range(3):
            _axpy(f[6 + i, 6 + c], P[o + 6 + c], row); _axpy(g[6 + i, c], P[bi + 3 + c], row)
    P[o + 6:o + 9] = scratch


@njit(cache=False)
def phi_vec(x, N, bo, dt, fs, As, Gs):
    """Block Phi times a vector."""
    out = x.copy()
    for i in range(N):
        o = 9 * i; bi = bo + 6 * i; f = fs[i]; a = As[i]; g = Gs[i]
        for e in range(3):
            t1 = x[o + e] + dt * x[o + 3 + e]; t2 = x[o + 3 + e]; t3 = 0.0
            for c in range(3):
                t1 += f[e, 6 + c] * x[o + 6 + c] + a[e, c] * x[bi + c]
                t2 += f[3 + e, 6 + c] * x[o + 6 + c] + a[3 + e, c] * x[bi + c]
                t3 += f[6 + e, 6 + c] * x[o + 6 + c] + g[6 + e, c] * x[bi + 3 + c]
            out[o + e] = t1; out[o + 3 + e] = t2; out[o + 6 + e] = t3
    return out


@njit(cache=False)
def _transpose_into(P, W):
    n = P.shape[0]
    for i in range(n):
        wi = W[i]
        for j in range(n):
            wi[j] = P[j, i]


@njit(cache=False)
def struct_core(acc, gyr, dt, g_vec, pivot, r, v, q, L, b, P, qk, qb, rd, cols, nnz, est, oc, joseph):
    T = acc.shape[0]; N = acc.shape[1]; nl = 2 * N - 1; n = P.shape[0]; m = 6 * N; nm = 9 * N
    lo = nm if est else -1; bo = nm + (3 * nl if est else 0)
    hr = np.zeros((T, N, 3)); hv = np.zeros((T, N, 3)); hq = np.zeros((T, N, 4)); hb = np.zeros((T, 6 * N))
    hL = np.zeros((T if est else 1, nl, 3)); w = np.zeros((N, 3))
    W = np.zeros((n, n)); C = np.zeros((m, n)); D = np.zeros((n, m)); S = np.zeros((m, m)); K = np.zeros((n, m))
    y = np.zeros(m); vals = np.zeros(cols.shape); Rs = np.zeros((N, 3, 3)); s3 = np.zeros((3, 3)); scratch = np.zeros((3, n))
    fs = np.zeros((N, 9, 9)); As = np.zeros((N, 9, 3)); Gs = np.zeros((N, 9, 3)); Js = np.zeros((N, 3, 3))
    other = np.array([i for i in range(n) if not (i < nm and i % 9 >= 6)], dtype=np.int64)
    z_prev = heading_dir(r, v, q, pivot, g_vec, n); z_new = z_prev.copy()
    avec = np.zeros(n); hv2 = np.zeros(n); hz = np.zeros(m); e = np.zeros(n)
    for k in range(T):
        if k > 0:
            p = k - 1
            for i in range(N):
                r2, v2, q2, f, a, g = prop_bias(r[i], v[i], q[i], acc[p, i], gyr[p, i], acc[k, i], gyr[k, i], b[6 * i:6 * i + 3], b[6 * i + 3:6 * i + 6], dt, g_vec)
                r[i] = r2; v[i] = v2; q[i] = q2; fs[i] = f; As[i] = a; Gs[i] = g
            if oc:
                z_new = heading_dir(r, v, q, pivot, g_vec, n)
                avec = (phi_vec(z_prev, N, bo, dt, fs, As, Gs) - z_new) / (z_prev @ z_prev)
                wv = np.zeros(n)
                for i in range(nm):
                    if z_prev[i] != 0.0: _axpy(z_prev[i], P[i], wv)
                hv2 = phi_vec(wv, N, bo, dt, fs, As, Gs) - .5 * (z_prev @ wv) * avec
            for i in range(N): phi_rows(P, 9 * i, bo + 6 * i, dt, fs[i], As[i], Gs[i], scratch)
            _transpose_into(P, W); P, W = W, P
            for i in range(N): phi_rows(P, 9 * i, bo + 6 * i, dt, fs[i], As[i], Gs[i], scratch)
            if oc:                       # Phi_oc = Phi - a z^T: P <- Phi P Phi^T - h a^T - a h^T
                for i in range(nm):
                    if avec[i] != 0.0: _axpy(-avec[i], hv2, P[i])
                for i in range(n):
                    pi = P[i]; hi = hv2[i]
                    for j in range(nm):
                        pi[j] -= hi * avec[j]
            for i in range(N):
                o = 9 * i; bi = bo + 6 * i
                for a_ in range(9):
                    for c in range(9):
                        P[o + a_, o + c] += qk[a_, c]
                for a_ in range(6): P[bi + a_, bi + a_] += qb[a_]
            for i in range(N): w[i] = gyr[k, i] - b[6 * i + 3:6 * i + 6]
            fill_H(N, est, r, v, q, L, w, pivot, y, vals, Rs, s3)
            apply_H(cols, nnz, vals, P, C)
            if oc:                       # H_oc = H - h e^T
                hz = np.zeros(m)
                for row in range(m):
                    t = 0.0
                    for idx in range(nnz[row]): t += vals[row, idx] * z_new[cols[row, idx]]
                    hz[row] = t
                e = z_new / (z_new @ z_new); ell = np.zeros(n)
                for i in range(nm):
                    if e[i] != 0.0: _axpy(e[i], P[i], ell)
                for row in range(m): _axpy(-hz[row], ell, C[row])
            for i in range(n):
                di = D[i]
                for row in range(m): di[row] = C[row, i]
            apply_H(cols, nnz, vals, D, S)
            if oc:
                for row in range(m):
                    t = 0.0
                    for i in range(nm): t += C[row, i] * e[i]
                    for i in range(m): S[i, row] -= hz[i] * t
            for i in range(m): S[i, i] += rd[i]
            Kt = np.linalg.solve(S.T, C)
            for i in range(n):
                ki = K[i]
                for row in range(m): ki[row] = Kt[row, i]
            dx = K @ (-y)
            for i in range(N):
                o = 9 * i
                r[i] += dx[o:o + 3]; v[i] += dx[o + 3:o + 6]
                qn = quat_mult_numba(q[i], _exp_quat_numba(dx[o + 6:o + 9])); qn /= np.sqrt(np.sum(qn ** 2)); q[i] = qn
                Js[i] = right_jacobian(dx[o + 6:o + 9])
            if est:
                for i in range(nl): L[i] += dx[lo + 3 * i:lo + 3 * i + 3]
            b += dx[bo:bo + 6 * N]
            KC = K @ C
            if joseph:
                KSK = (K @ S) @ Kt
                for i in range(n):
                    for j in range(i, n):
                        t = P[i, j] + P[j, i]
                        t = .5 * t + .5 * (KSK[i, j] + KSK[j, i]) - KC[i, j] - KC[j, i]
                        P[i, j] = t; P[j, i] = t
            else:
                for i in range(n):
                    for j in range(i, n):
                        t = .5 * (P[i, j] + P[j, i] - KC[i, j] - KC[j, i])
                        P[i, j] = t; P[j, i] = t
            for i in range(N):           # attitude reset: rows, then columns by symmetry, attitude blocks explicitly
                o = 9 * i + 6; J = Js[i]
                for a_ in range(3):
                    row = scratch[a_]; row[:] = 0.0
                    for c in range(3): _axpy(J[a_, c], P[o + c], row)
                P[o:o + 3] = scratch
            for i in range(N):
                o = 9 * i + 6
                for idx in range(other.shape[0]):
                    j = other[idx]
                    P[j, o] = P[o, j]; P[j, o + 1] = P[o + 1, j]; P[j, o + 2] = P[o + 2, j]
            for i in range(N):
                oi = 9 * i + 6
                for l in range(i, N):
                    ol = 9 * l + 6; J = Js[l]
                    for a_ in range(3):
                        t0 = P[oi + a_, ol] * J[0, 0] + P[oi + a_, ol + 1] * J[0, 1] + P[oi + a_, ol + 2] * J[0, 2]
                        t1 = P[oi + a_, ol] * J[1, 0] + P[oi + a_, ol + 1] * J[1, 1] + P[oi + a_, ol + 2] * J[1, 2]
                        t2 = P[oi + a_, ol] * J[2, 0] + P[oi + a_, ol + 1] * J[2, 1] + P[oi + a_, ol + 2] * J[2, 2]
                        P[oi + a_, ol] = t0; P[oi + a_, ol + 1] = t1; P[oi + a_, ol + 2] = t2
                    if l > i:
                        for a_ in range(3):
                            for c in range(3): P[ol + c, oi + a_] = P[oi + a_, ol + c]
            if k % 1000 == 0 and (not np.isfinite(P).all() or not np.isfinite(dx).all()):
                raise FloatingPointError('non-finite coupled update')
            if oc:
                z_prev = z_new
        hr[k] = r; hv[k] = v; hq[k] = q; hb[k] = b
        if est: hL[k] = L
    return hr, hv, hq, hL, hb


def run(m, N, r0, v0, q0, L0, est, oc, lever_sd, motion_sd, joseph=True, pattern=None):
    dt = float(np.median(np.diff(m['time']))); acc, gyr = nc.pack(m, N)
    P = nc.initial_cov(N, est, lever_sd, motion_sd)
    qb = np.r_[np.full(3, nc.SBA ** 2 * dt), np.full(3, nc.SBG ** 2 * dt)]
    cols, nnz = h_pattern(N, est) if pattern is None else pattern
    hr, hv, hq, hL, hb = struct_core(acc, gyr, dt, nc.G_VEC, nc.pivot(N), r0.copy(), v0.copy(), q0.copy(), np.array(L0, float).copy(),
                                     np.zeros(6 * N), P, nc.QK, qb, nc.rdiag(N), cols, nnz, bool(est), bool(oc), bool(joseph))
    return dict(r=hr, v=hv, q=hq, L=hL, b=hb)
