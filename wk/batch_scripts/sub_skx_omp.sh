#!/bin/sh
#SBATCH -J omp               # jobname
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of processes per node
#SBATCH --cpus-per-task=40   # logical core per task
#SBATCH --time=00:30:00      # execute time (hh:mm:ss)
#SBATCH --account=TBTOK      # account number
#SBATCH -o %j.out            # strout filename (%j is jobid)
#SBATCH -e %j.err            # stderr filename (%j is jobid)
#SBATCH -p dev               # Job class
 
source /opt/modules/default/init/bash
module switch PrgEnv-intel PrgEnv-gnu
module unload cray-libsci/18.04.1
module load intel
module load cray-fftw
 
export OMP_NUM_THREADS=40
export OMP_PROC_BIND=true
 
srun ./vlp4d.skx_omp SLD10.dat
