"""Analytical slider kinematics and seeded IMU measurements."""
import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_slider_crank_trajectory_v3(
    L1=2.0,      # crank length (m)
    L2=4.0,      # connecting-rod length (m)
    omega=2.0,   # crank angular velocity (rad/s)
    duration=10.0,
    dt=0.005,    # 200 Hz
    # --- noise and bias parameters ---
    noise_std_acc=0.01,    # accelerometer white-noise SD (m/s^2), about 1 mg
    noise_std_gyr=0.001,   # gyroscope white-noise SD (rad/s), about 0.06 deg/s
    bias_acc_range=0.02,   # accelerometer bias bound (m/s^2), about 2 mg
    bias_gyr_range=0.002    # gyroscope bias bound (rad/s), about 0.1 deg/s
):
    """
    IMU data of the crank, rod and slider, with constant bias and white noise.
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    g_vec = np.array([0, -9.81, 0])  # gravity vector
    
    # --- 1. Kinematics ---
    theta1 = omega * t
    omega1 = np.full(N, omega)
    sin_th2 = -(L1/L2) * np.sin(theta1)
    theta2 = np.arcsin(sin_th2)
    cos_th1, cos_th2 = np.cos(theta1), np.cos(theta2)
    omega2 = - (L1 * cos_th1 * omega1) / (L2 * cos_th2)
    term1 = L1 * np.sin(theta1) * (omega1**2)
    term2 = L2 * np.sin(theta2) * (omega2**2)
    alpha2 = (term1 + term2) / (L2 * cos_th2)

    # --- 2. IMU 1 (Crank Center) ---
    r1 = L1 / 2.0
    p1 = np.column_stack((r1 * cos_th1, r1 * np.sin(theta1), np.zeros(N)))
    v1 = np.column_stack((-r1 * omega1 * np.sin(theta1), r1 * omega1 * cos_th1, np.zeros(N)))
    a1 = np.column_stack((-r1 * omega1**2 * cos_th1, -r1 * omega1**2 * np.sin(theta1), np.zeros(N)))

    # --- 3. IMU 2 (Rod Center) ---
    r2 = L2 / 2.0
    pos_j1 = np.column_stack((L1 * cos_th1, L1 * np.sin(theta1), np.zeros(N)))
    vel_j1 = np.column_stack((-L1 * omega1 * np.sin(theta1), L1 * omega1 * cos_th1, np.zeros(N)))
    acc_j1 = np.column_stack((-L1 * omega1**2 * cos_th1, -L1 * omega1**2 * np.sin(theta1), np.zeros(N)))
    
    rel_pos2 = np.column_stack((r2 * cos_th2, r2 * np.sin(theta2), np.zeros(N)))
    rel_vel2 = np.column_stack((-r2 * omega2 * np.sin(theta2), r2 * omega2 * cos_th2, np.zeros(N)))
    rel_acc2 = np.column_stack((
        -r2 * alpha2 * np.sin(theta2) - r2 * omega2**2 * cos_th2,
         r2 * alpha2 * cos_th2 - r2 * omega2**2 * np.sin(theta2),
         np.zeros(N)
    ))
    p2 = pos_j1 + rel_pos2
    v2 = vel_j1 + rel_vel2
    a2 = acc_j1 + rel_acc2

    # --- 4. IMU 3 (Slider / Piston Center) ---
    slider_x = L1 * cos_th1 + L2 * cos_th2
    p3 = np.column_stack((slider_x, np.zeros(N), np.zeros(N)))
    v3_x = -L1 * omega1 * np.sin(theta1) - L2 * omega2 * np.sin(theta2)
    v3 = np.column_stack((v3_x, np.zeros(N), np.zeros(N)))
    a3_x = -L1 * (omega1**2 * cos_th1) - L2 * (alpha2 * np.sin(theta2) + omega2**2 * cos_th2)
    a3 = np.column_stack((a3_x, np.zeros(N), np.zeros(N)))

    # --- 5. Noise and output ---
    def apply_imu_noise(acc_clean, gyr_clean, seed=42):
        """Add a constant bias and white noise to one IMU (independent draws per IMU)."""
        # random constant bias, one per axis
        np.random.seed(seed)
        b_acc = np.random.uniform(-bias_acc_range, bias_acc_range, 3)
        b_gyr = np.random.uniform(-bias_gyr_range, bias_gyr_range, 3)
        
        # white noise
        n_acc = np.random.normal(0, noise_std_acc, acc_clean.shape)
        n_gyr = np.random.normal(0, noise_std_gyr, gyr_clean.shape)
        
        # measurement = truth + bias + noise
        acc_noisy = acc_clean + b_acc + n_acc
        gyr_noisy = gyr_clean + b_gyr + n_gyr
        
        return acc_noisy, gyr_noisy

    def get_imu_data(acc_world, vel_world, pos_world, gyro_z_val, theta, seed=42):
        rot = R.from_euler('z', theta)
        acc_body_clean = np.zeros((N, 3))
        gyr_body_clean = np.column_stack((np.zeros(N), np.zeros(N), gyro_z_val))
        
        for i in range(N):
            # true specific force = R.inv() @ (a_world - g_world)
            acc_body_clean[i] = rot[i].inv().apply(acc_world[i] - g_vec)
            
        # add bias and noise
        acc_body_noisy, gyr_body_noisy = apply_imu_noise(acc_body_clean, gyr_body_clean, seed=seed)
        
        return {
            'acc': acc_body_noisy, 
            'gyr': gyr_body_noisy,
            'quat': rot.as_quat(),
            'pos': pos_world,
            'vel': vel_world
        }

    imu1_data = get_imu_data(a1, v1, p1, omega1, theta1, seed=1)
    imu2_data = get_imu_data(a2, v2, p2, omega2, theta2, seed=2)
    imu3_data = get_imu_data(a3, v3, p3, np.zeros(N), np.zeros(N), seed=3)

    return {
        'time': t,
        'imu1': imu1_data,
        'imu2': imu2_data,
        'imu3': imu3_data
    }
