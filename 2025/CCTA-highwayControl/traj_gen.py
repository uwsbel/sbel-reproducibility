import numpy as np

def generate_ellipse_trajectory(a, b, num_points, center_x=0, center_y=0):
    """
    Generate an ellipse trajectory with x, y coordinates and heading.
    
    :param a: Semi-major axis of the ellipse
    :param b: Semi-minor axis of the ellipse
    :param num_points: Number of points to generate along the trajectory
    :param center_x: X-coordinate of the ellipse center (default: 0)
    :param center_y: Y-coordinate of the ellipse center (default: 0)
    :return: List of tuples (x, y, heading) for each point on the trajectory
    """
    t = np.linspace(0, 2*np.pi, num_points)
    x = center_x + a * np.cos(t)
    y = center_y + b * np.sin(t)
    
    # Calculate heading using atan2(delta_y, delta_x)
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    heading = np.arctan2(dy, dx)
    
    return list(zip(x, y, heading))

def generate_synthetic_trajectory(length_of_trajectory):
    # define key points of the trajectory
    keypoints = np.array([
        [-500, 0],
        [-400, 0],
        [-300, 0],
        [-200, 0],
        [-100, 0],
        [200, 0],
        [300, 0],
        [500, 0],
        [600, 0],
        [800, 0]
    ])
    
    # Calculate cumulative distances between keypoints
    distances = np.sqrt(np.sum(np.diff(keypoints, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    from scipy.interpolate import CubicSpline
    # Create cubic spline functions for x and y
    cs_x = CubicSpline(cumulative_distances, keypoints[:, 0])
    cs_y = CubicSpline(cumulative_distances, keypoints[:, 1])
    
    # Generate equally spaced points along the trajectory
    t = np.linspace(0, cumulative_distances[-1], length_of_trajectory)
    x = cs_x(t)
    y = cs_y(t)
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    heading = np.arctan2(dy, dx )
    # Combine x and y into a single array of points
    trajectory = np.vstack((x, y, heading)).T
    return trajectory, keypoints
# Example usage
if __name__ == "__main__":
    trajectory, keypoints = generate_synthetic_trajectory(10000)
    
    # # Print the first few points of the trajectory
    # for i, (x, y, heading) in enumerate(trajectory[:5]):
    #         print(f"Point {i}: x={x:.2f}, y={y:.2f}, heading={np.degrees(heading):.2f} degrees")
    # save the trajectory to a file
    # np.savetxt('trajectory_complex.csv', trajectory, delimiter=',',fmt='%f')
    np.savetxt('trajectory_straight.csv', trajectory, delimiter=',',fmt='%f')
    # Visualize the trajectory
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start Point')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', label='End Point')
    plt.plot(keypoints[:, 0], keypoints[:, 1], 'kx', label='Keypoints')
    plt.legend()
    plt.title('Synthetic Trajectory through Keypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()