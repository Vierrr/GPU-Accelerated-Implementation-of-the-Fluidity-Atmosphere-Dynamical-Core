# GPU-Accelerated Extension of Fluidity-Atmosphere

This repository provides the GPU-parallel implementation of the 
Fluidity-Atmosphere dynamical core.

The implementation is designed as a source-level extension of the 
Fluidity model. It requires replacing specific source files 
in an official Fluidity release prior to compilation.

------------------------------------------------------------
### 1. SOFTWARE REQUIREMENTS
------------------------------------------------------------

The implementation was tested in the following environment:

**GPU: NVIDIA A100**  
- FP64 peak performance: 9.7 TFLOPS  
- Maximum clock frequency: 1410 MHz  
- 108 SMs  
- L1 cache: 192 KiB  
- L2 cache: 40960 KiB  
- Register file per SM: 256 KiB  
- PCIe bandwidth: 32 GB/s  
- CUDA compiler: nvcc 13.0  

**CPU: AMD EPYC 7713 64-Core Processor**  
- ISA: x86_64  
- 64 cores  
- Maximum clock frequency: 3720 MHz  
- L1 cache: 8 MiB  
- L2 cache: 64 MiB  

**Compilers and tools**:
- gcc 13.3  
- gfortran 13.3  
- MPICH 3.4.2  
- CMake 3.22.1  

**Optimization flags**:
- O3  
- ffast-math  

Other Linux environments may work but are not officially tested.

------------------------------------------------------------
### 2. OBTAIN ORIGINAL FLUIDITY
------------------------------------------------------------

Fluidity is distributed under the **GNU Lesser General Public License (LGPL)**.

Download the official release from:

https://fluidityproject.github.io/get-fluidity.html

The version used in this study is:

**Fluidity version-2025.12**

Example:

`git clone https://github.com/FluidityProject/fluidity.git`

------------------------------------------------------------
### 3. APPLY GPU EXTENSION
------------------------------------------------------------

This repository contains modified source files and GPU-specific kernels.

**Step 1**: Copy the following modified files into the corresponding 
locations in the Fluidity source tree, replacing the original files:

`Advection_Diffusion_CG.F90`  
`Compressible_Projection.F90`  
`Divergence_Matrix_CG.F90`  
`Fields_Allocates.F90`  
`Momentum_CG.F90`  
`Transform_elements.F90`  

**Step 2**: Copy the entire `gpu` directory into the root directory 
of the Fluidity source tree.

------------------------------------------------------------
### 4. COMPILING
------------------------------------------------------------

Load the Fluidity environment:

`source fluidity.env`

Use provided compilation script

`cd gpu`         
`bash gpu_compiling.sh`

Compile:

`make -j1`

------------------------------------------------------------
### 5. RUNNING SIMULATIONS
------------------------------------------------------------

Simulations are executed using the standard Fluidity interface.

Example:

`run xxx/bin/fluidity -v3 -l ./MountainWave_3D.flml`

Expected outputs:

- Log files containing timing statistics  
- Performance metrics printed to standard output  

The **test data** used in this study are publicly available at:

https://doi.org/10.5281/zenodo.17824052
