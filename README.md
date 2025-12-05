# GPU-Accelerated-Implementation-of-the-Fluidity-Atmosphere-Dynamical-Core
This repository provides the GPU-accelerated implementation of the Fluidity-Atmosphere dynamical core developed in our study. It includes the core Fortran modules, CUDA kernels, GPU–CPU interface code, and build scripts necessary to reproduce the GPU results reported in the paper.
# Repository Structure

The repository consists of three major parts:

1. Core Fortran modules
Implements the dynamical core components that interface with GPU routines.

2. CUDA kernels and GPU interface code
Contains GPU-optimized kernels, data structure definitions, and CUDA–Fortran interface wrappers.

3. Build scripts and utility files
Includes the compilation script and auxiliary source files for initializing and managing GPU execution.

# Build
```bash
bash gpu_compiling.sh
```

This script compiles the CUDA kernels and links the GPU routines with the Fluidity solver.

# Usage

After building, the executable uses GPU kernels for element integration, matrix assembly, and key solvers within the Fluidity-Atmosphere workflow.

# Citation

If using this code in academic work, please cite the associated paper and the Fluidity model.
