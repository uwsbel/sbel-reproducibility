import numpy as np
import pandas as pd
import sympy as sm
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace

def calculate_quaternions(theta, phi):
    """
    Converts Z(phi) -> X(theta) Euler sequence to Quaternion (w, x, y, z).
    """
    # Half angles
    c_phi = np.cos(phi / 2.0)
    s_phi = np.sin(phi / 2.0)
    c_theta = np.cos(theta / 2.0)
    s_theta = np.sin(theta / 2.0)
    
    # Formula for Z -> X rotation
    w = c_phi * c_theta
    x = c_phi * s_theta
    y = s_phi * s_theta
    z = s_phi * c_theta
    
    return np.array([w, x, y, z])

def run_simulation_best_practice(duration=20.0):
    print("========================================================")
    print("1. INITIALIZATION & SYMBOLIC DERIVATION")
    print("   (Using SymPy to derive equations. Please wait 10-20s...)")
    print("========================================================")

    # ---------------------------------------------------------
    # A. Define Symbols & Generalized Coordinates
    # ---------------------------------------------------------
    m1, m2, L1, L2, g, t = sm.symbols('m1 m2 L1 L2 g t')
    
    # q1, q3: Theta (Angle from vertical -Z axis)
    # q2, q4: Phi   (Azimuthal rotation around Z axis)
    q1, q2, q3, q4 = me.dynamicsymbols('theta1 phi1 theta2 phi2')
    q1d, q2d, q3d, q4d = me.dynamicsymbols('theta1 phi1 theta2 phi2', 1)
    q1dd, q2dd, q3dd, q4dd = me.dynamicsymbols('theta1 phi1 theta2 phi2', 2)

    # ---------------------------------------------------------
    # B. Define Reference Frames
    # ---------------------------------------------------------
    # N: Inertial Global Frame (Z is Up)
    N = me.ReferenceFrame('N')
    
    # Body Frames (B1 & B2)
    # We define B.z to align with the rod direction.
    # Rotation Sequence: Z (Phi) -> X (Theta)
    B1 = N.orientnew('B1', 'Body', [q2, q1, 0], 'ZXY') 
    B2 = N.orientnew('B2', 'Body', [q4, q3, 0], 'ZXY')

    # ---------------------------------------------------------
    # C. Kinematics (Points & Velocities)
    # ---------------------------------------------------------
    O = me.Point('O')
    O.set_vel(N, 0)
    
    # -- Body 1 --
    # P1_COM: IMU Location (Midpoint of rod)
    # P1_END: Joint to Body 2 (End of rod)
    P1_COM = O.locatenew('P1_COM', - (L1/2) * B1.z)
    P1_END = O.locatenew('P1_END', - L1 * B1.z)
    
    # -- Body 2 --
    # P2_COM: IMU Location (Midpoint of rod)
    # P2_END: End of pendulum
    P2_COM = P1_END.locatenew('P2_COM', - (L2/2) * B2.z)
    P2_END = P1_END.locatenew('P2_END', - L2 * B2.z)

    # Calculate Velocities in Global Frame
    P1_COM.set_vel(N, P1_COM.pos_from(O).dt(N))
    P1_END.set_vel(N, P1_END.pos_from(O).dt(N))
    P2_COM.set_vel(N, P2_COM.pos_from(O).dt(N))
    P2_END.set_vel(N, P2_END.pos_from(O).dt(N))

    # ---------------------------------------------------------
    # D. Dynamics (Lagrangian)
    # ---------------------------------------------------------
    # Kinetic Energy (Assuming point masses at the ENDS of rods)
    vel1 = P1_END.vel(N)
    vel2 = P2_END.vel(N)
    
    # Note: Using .dot() for squared magnitude to be safe across versions
    T = 0.5 * m1 * vel1.dot(vel1) + 0.5 * m2 * vel2.dot(vel2)
    
    # Potential Energy (Gravity acts in -Z direction)
    V = m1 * g * P1_END.pos_from(O).dot(N.z) + m2 * g * P2_END.pos_from(O).dot(N.z)
    
    Lagrangian = T - V
    LM = me.LagrangesMethod(Lagrangian, [q1, q2, q3, q4])
    equations = LM.form_lagranges_equations()

    # ---------------------------------------------------------
    # E. Measurement Equations
    # ---------------------------------------------------------
    g_vector = -g * N.z
    
    # 1. Validation Data (Global Frame N)
    pos1_global = P1_COM.pos_from(O).to_matrix(N)
    vel1_global = P1_COM.vel(N).to_matrix(N)
    pos2_global = P2_COM.pos_from(O).to_matrix(N)
    vel2_global = P2_COM.vel(N).to_matrix(N)

    # 2. Sensor Data (Local Body Frame B)
    # Accelerometer: (a_inertial - g) projected to Body Frame
    imu1_acc_local = (P1_COM.acc(N) - g_vector).to_matrix(B1)
    imu2_acc_local = (P2_COM.acc(N) - g_vector).to_matrix(B2)
    
    # Gyroscope: Angular Velocity projected to Body Frame
    imu1_gyro_local = B1.ang_vel_in(N).to_matrix(B1)
    imu2_gyro_local = B2.ang_vel_in(N).to_matrix(B2)

    # ---------------------------------------------------------
    # F. Compile Functions (Lambdify)
    # ---------------------------------------------------------
    print("   -> Compiling numerical functions...")
    
    states = [q1, q2, q3, q4, q1d, q2d, q3d, q4d]
    controls = [q1dd, q2dd, q3dd, q4dd] 
    params = [m1, m2, L1, L2, g]
    
    # Dynamics (M * qdd = F)
    f_mass = sm.lambdify(states + params, LM.mass_matrix_full, "numpy")
    f_forcing = sm.lambdify(states + params, LM.forcing_full, "numpy")
    
    # Sensors (Requires q, qd, AND qdd)
    f_imu1_acc = sm.lambdify(states + controls + params, imu1_acc_local, "numpy")
    f_imu1_gyro = sm.lambdify(states + controls + params, imu1_gyro_local, "numpy")
    f_imu2_acc = sm.lambdify(states + controls + params, imu2_acc_local, "numpy")
    f_imu2_gyro = sm.lambdify(states + controls + params, imu2_gyro_local, "numpy")

    # Ground Truth (Requires q, qd)
    f_pos1_glob = sm.lambdify(states + params, pos1_global, "numpy")
    f_vel1_glob = sm.lambdify(states + params, vel1_global, "numpy")
    f_pos2_glob = sm.lambdify(states + params, pos2_global, "numpy")
    f_vel2_glob = sm.lambdify(states + params, vel2_global, "numpy")
    
    # Energy Calculation (for verification)
    f_total_energy = sm.lambdify(states + params, T + V, "numpy")

    print("   -> Derivation Complete.")

    # ========================================================
    # 2. SIMULATION (solve_ivp)
    # ========================================================
    print("\n========================================================")
    print("2. RUNNING SIMULATION (solve_ivp - Radau)")
    print("========================================================")
    
    # Parameters
    p_m1, p_m2 = 1.0, 1.0
    p_L1, p_L2 = 2.0, 2.0
    p_g  = 9.81
    ROOT_POS = np.array([0.0, 0.0, 4.0]) # Global Offset

    # Simulation Settings
    t_duration = float(duration)
    dt_output = 0.001
    t_eval = np.arange(0, t_duration, dt_output)

    # Initial State
    # [theta1, phi1, theta2, phi2, w1, w2, w3, w4]
    init_state = np.array([
        np.radians(45), 0.0,            
        np.radians(60), np.radians(90), 
        0.0, 0.0, 0.0, 0.0              
    ])

    # Wrapper for solve_ivp (t must be the first argument)
    def dynamics_ivp(t, state, m1, m2, L1, L2, g):
        M = f_mass(*state, m1, m2, L1, L2, g)
        F = f_forcing(*state, m1, m2, L1, L2, g)
        dq_dt = np.linalg.solve(M, F)
        return dq_dt.flatten()

    print("   -> Solving Stiff ODE...")
    start_time = time.time()
    
    # USE 'Radau': Excellent for stiff systems (energy conservation)
    # Integrate in 10 s blocks, so the first block does not depend on the record length.
    assert t_duration in (10.,20.)
    pieces=[];state=init_state
    for start in np.arange(0.,t_duration,10.):
        end=start+10.
        times=t_eval[(t_eval>=start)&(t_eval<end)]
        part=solve_ivp(fun=dynamics_ivp,t_span=[start,end],y0=state,
            args=(p_m1,p_m2,p_L1,p_L2,p_g),method='Radau',
            t_eval=np.r_[times,end],rtol=1e-12,atol=1e-12)
        if not part.success:raise RuntimeError(part.message)
        state=part.y[:,-1].copy()
        pieces.append(part.y[:,:-1])
    sol=SimpleNamespace(y=np.concatenate(pieces,axis=1),t=t_eval)

    print(f"   -> Solved in {time.time() - start_time:.2f} seconds.")
    
    # Transpose result to shape (N_samples, N_states)
    state_history = sol.y.T 
    time_array = sol.t

    # ========================================================
    # 3. DATA GENERATION
    # ========================================================
    print("\n========================================================")
    print("3. GENERATING SENSOR & TRUTH DATA")
    print("========================================================")
    
    data_rows = []
    energy_log = []

    for i, t_val in enumerate(time_array):
        curr_s = state_history[i]
        
        # A. Calculate Energy for Verification
        E_now = f_total_energy(*curr_s, p_m1, p_m2, p_L1, p_L2, p_g)
        energy_log.append(E_now)
        
        # B. Solve for Acceleration (qdd)
        # We need this because solve_ivp only gives us q and qd.
        # Sensors (Accel) need F=ma.
        M_now = f_mass(*curr_s, p_m1, p_m2, p_L1, p_L2, p_g)
        F_now = f_forcing(*curr_s, p_m1, p_m2, p_L1, p_L2, p_g)
        derivs = np.linalg.solve(M_now, F_now).flatten()
        qdd_now = derivs[4:] 
        
        # Argument lists
        kin_args = list(curr_s) + [p_m1, p_m2, p_L1, p_L2, p_g]
        sen_args = list(curr_s) + list(qdd_now) + [p_m1, p_m2, p_L1, p_L2, p_g]
        
        # C. Compute Values
        # 1. Global Truth (Pos/Vel)
        pos1 = f_pos1_glob(*kin_args).flatten() + ROOT_POS
        vel1 = f_vel1_glob(*kin_args).flatten()
        pos2 = f_pos2_glob(*kin_args).flatten() + ROOT_POS
        vel2 = f_vel2_glob(*kin_args).flatten()
        quat1 = calculate_quaternions(curr_s[0], curr_s[1]) 
        quat2 = calculate_quaternions(curr_s[2], curr_s[3])

        
        # 2. Local Sensors (Acc/Gyro)
        acc1 = f_imu1_acc(*sen_args).flatten()
        gyro1 = f_imu1_gyro(*sen_args).flatten()
        acc2 = f_imu2_acc(*sen_args).flatten()
        gyro2 = f_imu2_gyro(*sen_args).flatten()
        
        # D. Append Data
        row = np.concatenate([
            [t_val],
            pos1, vel1, acc1, gyro1, [quat1[0], quat1[1], quat1[2], quat1[3]],  # Body 1
            pos2, vel2, acc2, gyro2, [quat2[0], quat2[1], quat2[2], quat2[3]]   # Body 2
        ])
        data_rows.append(row)

    # Save to DataFrame
    cols = [
        'Time_s',
        'IMU1_Pos_X', 'IMU1_Pos_Y', 'IMU1_Pos_Z', 'IMU1_Vel_X', 'IMU1_Vel_Y', 'IMU1_Vel_Z',
        'IMU1_Acc_X', 'IMU1_Acc_Y', 'IMU1_Acc_Z', 'IMU1_Gyro_X', 'IMU1_Gyro_Y', 'IMU1_Gyro_Z',
        'IMU1_Quat_W', 'IMU1_Quat_X', 'IMU1_Quat_Y', 'IMU1_Quat_Z',
        'IMU2_Pos_X', 'IMU2_Pos_Y', 'IMU2_Pos_Z', 'IMU2_Vel_X', 'IMU2_Vel_Y', 'IMU2_Vel_Z',
        'IMU2_Acc_X', 'IMU2_Acc_Y', 'IMU2_Acc_Z', 'IMU2_Gyro_X', 'IMU2_Gyro_Y', 'IMU2_Gyro_Z',
        'IMU2_Quat_W', 'IMU2_Quat_X', 'IMU2_Quat_Y', 'IMU2_Quat_Z'
    ]
    df = pd.DataFrame(data_rows, columns=cols)
    
    return df, dict(relative_energy_range=float((np.max(energy_log)-np.min(energy_log))/abs(np.mean(energy_log))))
