"""
Plot cart-pole control results
Shows 5 variables (θ, x, θ̇, ẋ, u) over 10 seconds
Baseline converges to zero, FNODE may diverge
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_dir):
    """Load simulation results from directory"""
    # Get script directory to construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_results_dir = os.path.join(script_dir, results_dir)

    if not os.path.exists(full_results_dir):
        print(f"Results directory {full_results_dir} not found")
        return None, None, None, None

    try:
        times = np.loadtxt(os.path.join(full_results_dir, "times.txt"))
        states = np.loadtxt(os.path.join(full_results_dir, "states.txt"))
        controls = np.loadtxt(os.path.join(full_results_dir, "controls.txt"))
        
        # Filter out any potential empty entries
        times = times[~np.isnan(times)] if times.size > 0 else times
        controls = controls[~np.isnan(controls)] if controls.size > 0 else controls
        
        print(f"Loaded {results_dir}: times={len(times)}, states={states.shape}, controls={len(controls)}")
        
        # Determine method name based on directory
        if "fnode" in results_dir.lower():
            method_name = "FNODE"
        elif "baseline" in results_dir.lower():
            method_name = "Analytical"
        else:
            # Fallback: read from summary
            method_name = "Unknown"
            summary_file = os.path.join(full_results_dir, "summary.txt")
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    first_line = f.readline().strip()
                    if "FNODE" in first_line:
                        method_name = "FNODE"
                    elif "Analytical" in first_line:
                        method_name = "Analytical"
        
        return times, states, controls, method_name
    except Exception as e:
        print(f"Error loading results from {results_dir}: {e}")
        return None, None, None, None

def create_control_plot():
    """Create cart-pole control plots"""
    
    # Load results from both methods
    fnode_data = load_results("results_fnode")
    baseline_data = load_results("results_baseline")
    
    if fnode_data[0] is None and baseline_data[0] is None:
        print("No results found. Please run simulations first.")
        return
    
    # Set up the plot (5 subplots for 5 variables as in Fig.23)
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    fig.suptitle('Cart-Pole Control Results (10 Second Simulation)', fontsize=14, fontweight='bold')
    
    # Variable labels and units
    variables = [
        (r'$\theta$', 'Pole Angle'),
        (r'$x$', 'Cart Position'), 
        (r'$\dot{\theta}$', 'Pole Angular Velocity'),
        (r'$\dot{x}$', 'Cart Velocity'),
        (r'$u$', 'Control Force')
    ]
    
    # Colors for different methods
    colors = {'FNODE': 'red', 'Analytical': 'blue', 'Unknown': 'gray'}
    linestyles = {'FNODE': ':', 'Analytical': '-', 'Unknown': '--'}
    
    # Plot each variable
    for i, (ylabel, title) in enumerate(variables):
        ax = axes[i]
        
        # Plot baseline results first (so FNODE appears on top)
        if baseline_data[0] is not None:
            times_b, states_b, controls_b, method_b = baseline_data
            if i < 4:  # State variables
                ax.plot(times_b, states_b[:, i],
                       color=colors.get(method_b, 'blue'),
                       linestyle=linestyles.get(method_b, '-'),
                       linewidth=2, label=f'{method_b}')
            else:  # Control variable
                # Handle different control array lengths
                print(f"Baseline plotting: times={len(times_b)}, controls={len(controls_b)}")
                if len(controls_b) == len(times_b):
                    # Same length - use as is
                    times_u = times_b
                    controls_u = controls_b
                elif len(controls_b) == len(times_b) - 1:
                    # MPC case - controls have one fewer element
                    times_u = times_b[:-1]  # Use times without last point
                    controls_u = controls_b
                else:
                    # Unexpected case - just use available data
                    min_len = min(len(times_b), len(controls_b))
                    times_u = times_b[:min_len]
                    controls_u = controls_b[:min_len]
                ax.plot(times_u, controls_u,
                       color=colors.get(method_b, 'blue'),
                       linestyle=linestyles.get(method_b, '-'),
                       linewidth=2, label=f'{method_b}')
        
        # Plot FNODE results last (so it appears on top)
        if fnode_data[0] is not None:
            times_f, states_f, controls_f, method_f = fnode_data
            if i < 4:  # State variables
                ax.plot(times_f, states_f[:, i], 
                       color=colors.get(method_f, 'red'), 
                       linestyle=linestyles.get(method_f, ':'),
                       linewidth=2, label=f'{method_f}', zorder=10)  # Higher zorder ensures it's on top
            else:  # Control variable
                # Handle different control array lengths
                print(f"FNODE plotting: times={len(times_f)}, controls={len(controls_f)}")
                if len(controls_f) == len(times_f):
                    # Same length - use as is
                    times_u = times_f
                    controls_u = controls_f
                elif len(controls_f) == len(times_f) - 1:
                    # MPC case - controls have one fewer element
                    times_u = times_f[:-1]  # Use times without last point
                    controls_u = controls_f
                else:
                    # Unexpected case - just use available data
                    min_len = min(len(times_f), len(controls_f))
                    times_u = times_f[:min_len]
                    controls_u = controls_f[:min_len]
                ax.plot(times_u, controls_u,
                       color=colors.get(method_f, 'red'),
                       linestyle=linestyles.get(method_f, ':'), 
                       linewidth=2, label=f'{method_f}', zorder=10)  # Higher zorder ensures it's on top
        
        # Formatting
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10.0)  # 10 seconds full simulation
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Set y-limits based on actual data range with padding
        # Collect all data for this variable
        all_data = []
        
        if i < 4:  # State variables
            if fnode_data[0] is not None:
                all_data.append(fnode_data[1][:, i])
            if baseline_data[0] is not None:
                all_data.append(baseline_data[1][:, i])
        else:  # Control variable
            if fnode_data[0] is not None:
                # Handle control array length
                if len(fnode_data[2]) == len(fnode_data[0]):
                    all_data.append(fnode_data[2])
                elif len(fnode_data[2]) == len(fnode_data[0]) - 1:
                    all_data.append(fnode_data[2])
                else:
                    all_data.append(fnode_data[2][:min(len(fnode_data[0]), len(fnode_data[2]))])
            if baseline_data[0] is not None:
                # Handle control array length
                if len(baseline_data[2]) == len(baseline_data[0]):
                    all_data.append(baseline_data[2])
                elif len(baseline_data[2]) == len(baseline_data[0]) - 1:
                    all_data.append(baseline_data[2])
                else:
                    all_data.append(baseline_data[2][:min(len(baseline_data[0]), len(baseline_data[2]))])
        
        # Calculate min and max with padding
        if all_data:
            all_values = np.concatenate(all_data)
            data_min = np.min(all_values)
            data_max = np.max(all_values)
            
            # Add padding of 3 units
            y_min = data_min - 1
            y_max = data_max + 1
            
            ax.set_ylim(y_min, y_max)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot to mpc_cartpole directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'cart_pole_control.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"Plot saved as: {plot_path}")

    # Close the plot
    plt.close()

def print_performance_comparison():
    """Print performance comparison between methods"""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (10 SECOND SIMULATION)")
    print("="*60)
    
    # Load and compare results
    fnode_data = load_results("results_fnode")
    baseline_data = load_results("results_baseline")
    
    methods = []
    if fnode_data[0] is not None:
        methods.append(("FNODE", fnode_data))
    if baseline_data[0] is not None:
        methods.append(("Baseline (Analytical)", baseline_data))
    
    for method_name, (times, states, controls, _) in methods:
        final_state = states[-1]
        final_error = np.linalg.norm(final_state)
        control_effort = np.sum(controls**2) * (times[1] - times[0]) if len(times) > 1 else 0
        max_control = np.max(np.abs(controls))
        
        # Check if solver hit limits
        hit_limit = np.any(np.abs(controls) > 99.99)
        
        print(f"\n{method_name} Results:")
        print(f"  Simulation duration: {times[-1]:.2f} seconds")
        print(f"  Final state: θ={final_state[0]:.4f}, x={final_state[1]:.4f}, "
              f"θ̇={final_state[2]:.4f}, ẋ={final_state[3]:.4f}")
        print(f"  Final error norm: {final_error:.6f}")
        print(f"  Control effort: {control_effort:.2f}")
        print(f"  Max |control|: {max_control:.2f} N")
        print(f"  Hit control limit: {'Yes ⚠️' if hit_limit else 'No'}")
        print(f"  Convergence: {'✓ Excellent' if final_error < 0.01 else '✗ Failed (diverged)' if final_error > 10 else '⚠ Poor'}")

def main():
    """Main function"""
    print("Generating cart-pole control plots for 10-second simulation...")
    print("Note: FNODE may diverge beyond plot limits after ~3 seconds")
    create_control_plot()
    print_performance_comparison()

if __name__ == "__main__":
    main()