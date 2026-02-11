import numpy as np
import matplotlib.pyplot as plt
 
"""
Slider-Crank -  Augmented-Lagrangian + Barzilai-Borwein
==========================================================
* Planar mechanism: 3 bodies (crank, rod, slider)
* DOF ordering  q = [x1, y1, th1,  x2, y2, th2,  x3, y3, th3]
* 8 scalar holonomic constraints
    C1 (2) crank-ground pin
    C2 (2) crank-rod pin
    C3 (2) rod-slider pin
    C4 (2) slider guide  (y₃ = 0, θ₃ = 0)
* Implicit **Backward Euler** for both velocities and positions.
* Inner solver: Barzilai-Borwein; outer: Augmented Lagrangian.
 
The script now includes **all non-inertial loads** found in the LaTeX write-up:
    ▸ driving motor torque   τ₁(t)
    ▸ non-smooth rotational damper  c₁₂‖ω₁−ω₂‖^γ sgn(ω₁−ω₂)
    ▸ slider Coulomb/viscous mix    c|ẋ₃|^ψ sgn(ẋ₃)
    ▸ nonlinear slider spring       K(ℓᵐᵃˣ − x₃)^δ  (active in compression)
"""
# -------------------------------------------------------------------
#  Physical parameters
# -------------------------------------------------------------------
# Masses [kg] and out‑of‑plane inertias [kg·m²]
m1, m2, m3 = 1.0, 1.0, 1.0
J1, J2, J3 = 0.10, 0.10, 0.10
 
# Half‑lengths  (full lengths are 2·L₁, 2·L₂) [m]
L1, L2 = 1.0, 2.0
 
# Gravity
g = 9.81  # [m/s²]
 
# Motor torque (user function)
tau1 = lambda t: -2 * np.sin(2 * np.pi * t)  # [N·m]
 
# Rotational damper between crank & rod
c12 = 0.05  # [N·m·(rad/s)^‑γ]
gamma = 1.0
 
# Slider friction + nonlinear spring
c_slide = 0.40  # [N·(m/s)^‑ψ]
psi = 1.0
K_spring = 1.0  # [N/m^δ]
delta = 1.0
x3_max = 2.5 * L1 + 2.0 * L2  # reference slider travel limit
 
# -------------------------------------------------------------------
#  Time‑stepping parameters
# -------------------------------------------------------------------
h = 1.0e-3  # step size [s]
T_final = 10.0  # total time [s]
N_steps = int(T_final / h)
 
n_gen = 9  # generalized coordinates
n_constr = 8  # constraints
 
M = np.diag([m1, m1, J1, m2, m2, J2, m3, m3, J3])
 
 
# -------------------------------------------------------------------
#  Constraint functions
# -------------------------------------------------------------------
 
def constraint(q: np.ndarray) -> np.ndarray:
    """
    c(q) = 0  (size 8).
    Here each value represents the violation of a specific
    holomonic constraint. The goal is to keep these values to be zero
    """
    x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
    return np.array([
        x1 - L1 * np.cos(th1),  # C1‑x
        y1 - L1 * np.sin(th1),  # C1‑y
        x1 + L1 * np.cos(th1) - x2 + L2 * np.cos(th2),  # C2‑x
        y1 + L1 * np.sin(th1) - y2 + L2 * np.sin(th2),  # C2‑y
        x2 + L2 * np.cos(th2) - x3,  # C3‑x
        y2 + L2 * np.sin(th2) - y3,  # C3‑y
        y3,  # C4‑y
        th3  # C4‑θ
    ])
 
 
def constraint_jacobian(q: np.ndarray) -> np.ndarray:
    x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
    J = np.zeros((8, 9))
 
    # C1
    J[0, 0] = 1.0
    J[0, 2] = L1 * np.sin(th1)
    J[1, 1] = 1.0
    J[1, 2] = -L1 * np.cos(th1)
 
    # C2
    J[2, 0] = 1.0
    J[2, 3] = -1.0
    J[2, 2] = -L1 * np.sin(th1)
    J[2, 5] = -L2 * np.sin(th2)
    J[3, 1] = 1.0
    J[3, 4] = -1.0
    J[3, 2] = L1 * np.cos(th1)
    J[3, 5] = L2 * np.cos(th2)
 
    # C3
    J[4, 3] = 1.0
    J[4, 6] = -1.0
    J[4, 5] = -L2 * np.sin(th2)
    J[5, 4] = 1.0
    J[5, 7] = -1.0
    J[5, 5] = L2 * np.cos(th2)
 
    # C4
    J[6, 7] = 1.0
    J[7, 8] = 1.0
 
    return J
 
 
# -------------------------------------------------------------------
#  Non‑inertial loads  f_ext(q, v, t)
# -------------------------------------------------------------------
 
def external_forces(q: np.ndarray, v: np.ndarray, lambda_val: float, t: float) -> np.ndarray:
    """Generalized loads (size 9) including gravity, drive, damper, spring."""
    x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
    vx1, vy1, w1, vx2, vy2, w2, vx3, vy3, w3 = v
 
    f = np.zeros(n_gen)
    # gravity
    f[1] = -m1 * g
    f[4] = -m2 * g
    f[7] = -m3 * g
    # motor torque + crank-rod damper (body-1 DOF θ₁)
    rel_w = w1 - w2
    damper = c12 * np.abs(rel_w) ** gamma * np.sign(rel_w)
    f[2] = tau1(t) - damper  # τ₁ - c₁₂ϕ̇
 
    # damper reaction on body-2 DOF θ₂
    f[5] = -damper  # ‑c₁₂ϕ̇  (opposite sign)
 
    # slider friction + spring (x‑direction of body‑3)
    # friction = c_slide * np.abs(vx3) ** psi * np.sign(vx3)
    friction = c_slide * abs(lambda_val) * np.sign(vx3)
    spring = 0.0
    comp = x3_max - x3  # only active when positive (compression)
    if comp > 0.0:
        spring = K_spring * comp ** delta
 
    f[6] = -(friction + spring)
 
    return f
 
 
# -------------------------------------------------------------------
#  ALM + BB velocity solve
# -------------------------------------------------------------------
 
def bb_alm_step(v_guess: np.ndarray, lam_guess: np.ndarray,
                v_prev: np.ndarray, q_prev: np.ndarray, t_next: float):
    v = v_guess.copy()
    lam = lam_guess.copy()
 
    rho = 1.0e10  # slightly milder than 1e10 for stability
    max_inner = 12
    max_outer = 25
    tol = 1.0e-6
    local_tol = 1.0e-1
    alpha = 1.0e-3
    use_bb1 = True
    normal_force = lam[6]
 
    def grad_L(v_loc: np.ndarray, lambda_val: float) -> np.ndarray:
        g_term = (M @ (v_loc - v_prev)) / h
        qA = q_prev + h * v_loc
        ext = external_forces(qA, v_loc, lambda_val,t_next)
        c_val = constraint(qA)
        J_val = constraint_jacobian(qA)
        return g_term - ext + J_val.T @ (lam + rho * h * c_val)
 
    for outer_iter in range(max_outer):
        local_tol = max(local_tol * 0.5, tol)
        vk, gk = v.copy(), grad_L(v, normal_force)
 
        for inner_iter in range(max_inner):
            vk1 = vk - alpha * gk
            gk1 = grad_L(vk1, lam[7])
            norm_gk1 = np.linalg.norm(gk1)
            # print inner loop statistics
            # print(f"inner {inner_iter}, norm(gk1)={norm_gk1:.2e}")
            if norm_gk1 < local_tol:
                vk, gk = vk1, gk1;
                break
            s, y = vk1 - vk, gk1 - gk
            if use_bb1:
                alpha = np.dot(s, s) / (np.dot(s, y) + 1e-12)
            else:
                alpha = np.dot(s, y) / (np.dot(y, y) + 1e-12)
            use_bb1 = not use_bb1
            vk, gk = vk1, gk1
 
        v = vk
        qA = q_prev + h * v
        c_val = constraint(qA)
        lam += rho * h * c_val
        normal_force = lam[6]
        # print outer loop statistics
        if outer_iter % 1000 == 0:
            print(f">>>>> End of  OUTER STEP #{outer_iter}; norm(constr_violation)={np.linalg.norm(c_val):.2e}")
 
        if np.linalg.norm(c_val) < tol:
            break
 
    return v, lam
 
 
# -------------------------------------------------------------------
#  Initial configuration: straight‑line pose
# -------------------------------------------------------------------
theta1_0 = theta2_0 = theta3_0 = 0.0
x1_0, y1_0 = L1, 0.0
x2_0, y2_0 = 2 * L1 + L2, 0.0
x3_0, y3_0 = 2 * L1 + 2 * L2, 0.0
 
q = np.array([x1_0, y1_0, theta1_0,
              x2_0, y2_0, theta2_0,
              x3_0, y3_0, theta3_0])
 
v = np.zeros(n_gen)
lam = np.zeros(n_constr)
v_guess = v.copy()
 
# -------------------------------------------------------------------
#  History arrays
# -------------------------------------------------------------------
q_hist = np.zeros((N_steps + 1, n_gen))
v_hist = np.zeros_like(q_hist)
a_hist = np.zeros_like(q_hist)  # Add acceleration history
q_hist[0] = q;
v_hist[0] = v
a_hist[0] = np.zeros(n_gen)  # Initial acceleration
 
# -------------------------------------------------------------------
#  Time integration loop
# -------------------------------------------------------------------
print("\n=== Slider-Crank ALM + BB (with damper & spring) ===")
for k in range(N_steps):
    t_next = (k + 1) * h
    v, lam = bb_alm_step(v_guess, lam, v_prev=v, q_prev=q, t_next=t_next)
 
    q += h * v  # position update
    a = (v - v_hist[k]) / h  # accel estimate
 
    q_hist[k + 1] = q
    v_hist[k + 1] = v
    a_hist[k + 1] = a  # Store acceleration
 
    v_guess = v + h * a  # Gustafson predictor
 
    if k % 1000 == 0:
        print(f"step {k:5d}  t={t_next: .3f}  |c|={np.linalg.norm(constraint(q)):.2e}")
 
# -------------------------------------------------------------------
#  Diagnostics plots
# -------------------------------------------------------------------
plt.figure()
plt.title("Crank time history")
plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 2], label="θ₁ [rad]")
plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 5], label="θ₂ [rad]")
plt.xlabel("time [s]")
plt.ylabel("θ₁ [rad], θ₂ [rad]")
plt.legend();
plt.grid(True)
plt.savefig("slider_crank_bb_output_theta.png", dpi=150, bbox_inches='tight')
 
plt.figure()
plt.plot(np.arange(N_steps + 1) * h, q_hist[:, 6], label="x₃(t)")
plt.xlabel("time [s]")
plt.ylabel("slider position x₃ [m]")
plt.legend()
plt.grid(True)
# Save the figure before closing
plt.savefig("slider_crank_bb_output.png", dpi=150, bbox_inches='tight')

# Add crank acceleration plot
plt.figure(figsize=(10, 6))
plt.title("Crank Angular Acceleration", fontsize=14, fontweight='bold')
plt.plot(np.arange(N_steps + 1) * h, a_hist[:, 2], label="α₁ [rad/s²]", linewidth=2.0, color='blue')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Angular Acceleration [rad/s²]", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("slider_crank_crank_acceleration.png", dpi=150, bbox_inches='tight')

plt.show()