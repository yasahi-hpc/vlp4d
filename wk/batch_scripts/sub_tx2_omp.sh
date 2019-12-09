#!/bin/sh
#SBATCH -p tx2
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -N 1
 
module load arm-compiler/19.2.0 gnu/8.3.0
module list
 
export OMP_NUM_THREADS=32
srun ./vlp4d.tx2_omp --kokkos-threads=32 SLD10_large.dat
