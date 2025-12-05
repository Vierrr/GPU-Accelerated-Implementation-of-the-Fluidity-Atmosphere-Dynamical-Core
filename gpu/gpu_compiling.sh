#!/bin/sh
echo "compiling gpu"
rm cuda_interface.o get_edge_lengths.o
nvcc --expt-relaxed-constexpr -arch=sm_80 -gencode arch=compute_80,code=compute_80 -O3 -lineinfo  --ptxas-options -warn-spills -arch sm_80 -c  cuda_interface.cu
nvcc -c --expt-relaxed-constexpr -arch=sm_80 -gencode arch=compute_80,code=compute_80 -O3 -lineinfo  --ptxas-options -warn-spills -arch sm_80 assemble_get_edge_lengths_stream.cu -o get_edge_lengths.o
