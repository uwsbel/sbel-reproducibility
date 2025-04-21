# Linear System Solvers

A collection of linear system solver implementations using various GPU and CPU backends for the paper titled "On the use of GPU-based linear solvers in Multibody Dynamics".

## Prerequisites

- CMake (>= 3.18)
- CUDA Toolkit with NVIDIA cuDSS (>= 0.5.0)
Pls refer to below link for installation of the CuDSS Library
https://docs.nvidia.com/cuda/cudss/getting_started.html#installation-and-compilation
- Intel oneMKL (for PARDISO solvers)
Pls refer to below link for installation of the oneMKL library
https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html

## Building the Project

1. Clone the repository (Note: Note: Ensure Git LFS is enabled to retrieve the matrix data)

2. Create and navigate to the build directory

3. Configure with CMake and build the project

## Running the Demos

The project includes the following executable demos:

### cuDSS Solver (GPU)

The GPU-based solver using NVIDIA's cuDSS library:

```bash
./task_cudss [num_spokes] [options]
```

Options:
- `num_spokes`: Number of spokes (default: 16, available: 16, 80)
- `-f, --float`: Use single precision
- `-d, --double`: Use double precision (default)

Example:
```bash
./task_cudss 80 --double
```

### PARDISO Solver (CPU)

The CPU-based solver using Intel's PARDISO:

```bash
./task_pardiso [num_threads] [num_spokes]
```
Where
- `num_threads` is the number of CPU threads to use.
- `num_spokes`: Number of spokes (default: 16, available: 16, 80)

Example:
```bash
./task_pardiso 16 80
```

## Data

The project includes sample data for testing in the `data/` directory:
- `data/ancf/16/`: Small test case (16 spokes)
- `data/ancf/80/`: Medium test case (80 spokes)
