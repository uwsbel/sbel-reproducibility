import numpy as np
import pychrono as chrono

def relative_state_to_goal(state, goal):
        x, y, theta, v = state
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        relative_pos = np.dot(rot_mat, np.array(goal) - np.array([x, y]))
        relative_theta = np.arctan2(relative_pos[1], relative_pos[0])
        return np.array([relative_pos[0], relative_pos[1], relative_theta, v], dtype=np.float32)

def wpts_path_plan(state, ref_traj,lookahead=0.0):
        """The Waypoints path planner.

        Computes the error state between the target state and the current state and the target state. First, get the target point which is closest ot the current state (plus the lookahead distance in the x direction). Then, get the error state by subtracting the current state from the target state (modular subtraction for the heading).
        Inputs:
             The state (x, y, heading, velocity) is vehicle's current state
             Reference trajectory is in format of n by 4, each row is (x_ref, y_ref, heading_ref, velocity_ref)
             Lookahead: look ahead distance in vehicle's heading angle direction
        Returns:
             The error state, and the target reference waypoint we are currently going to.

        """
        x_current = state[0]
        y_current = state[1]
        # theta_current = np.arctan2( 2*(x*w+z*y), x**2-w**2+y**2-z**2 )
        theta_current = state[2]
        while theta_current < -np.pi:
            theta_current = theta_current + 2 * np.pi
        while theta_current > np.pi:
            theta_current = theta_current - 2 * np.pi

        v_current = state[3]

        # get_logger().info("SPEED: "+str(v_current))

        dist = np.zeros((1, len(ref_traj[:, 1])))
        for i in range(len(ref_traj[:, 1])):
            dist[0][i] = dist[0][i] = (
                x_current + np.cos(theta_current) * lookahead - ref_traj[i][0]
            ) ** 2 + (
                y_current + np.sin(theta_current) * lookahead - ref_traj[i][1]
            ) ** 2
            index = dist.argmin()

        ref_state_current = list(ref_traj[index, :])
        ref_state_current[3] = 1.0
        return ref_state_current

def error_state(state, ref_traj, lookahead=1.0):
        x_current = state[0]
        y_current = state[1]
        theta_current = state[2]
        v_current = state[3]
        s_current = state[4]
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(ref_traj[:,1])))
        for i in range(len(ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*lookahead-ref_traj[i][0])**2+(y_current+np.sin(theta_current)*lookahead-ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, ref_state_current[3]-v_current, -s_current]
        return error_state

def SetRefTrajectory(filename):
    """Load the reference trajectory from a file and add visual shapes for the path in the Chrono system.

    Inputs:
        filename: Path to the CSV file containing the reference trajectory.
    Returns:
        ref_traj: The loaded reference trajectory.
    """
    # Load the reference trajectory from the CSV file
    ref_traj = np.genfromtxt(filename, delimiter=',')

    return ref_traj

def SetPathVisualization(ref_traj, msystem):
    # Create a visual material for the path
    vis_mat_path = chrono.ChVisualMaterial()
    vis_mat_path.SetDiffuseColor(chrono.ChColor(0.0, 1.0, 0.0))
    ref_traj = list(ref_traj)
    # visulize only every 10th point
    ref_traj = ref_traj[::2]
    # Add visual shapes for the path
    for pos in ref_traj:
        center_x, center_y, heading = pos[0], pos[1], pos[2]
        box_body = chrono.ChBodyEasyBox(0.06, 0.01, 0.01, 1000, True, False)
        #box_body = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, 0.05, 0.01, 50, True, False)
        box_body.SetPos(chrono.ChVector3d(center_x, center_y, 0.01))
        box_body.SetRot(chrono.QuatFromAngleZ(heading))
        box_body.SetFixed(True)
        # Set visual material for the box
        shape = box_body.GetVisualModel().GetShape(0)
        shape.AddMaterial(vis_mat_path)
        msystem.Add(box_body)

