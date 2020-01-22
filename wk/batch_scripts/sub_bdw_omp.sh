#!/bin/sh
#$ -cwd               # job execution in the current directory   
#$ -l f_node=1        # Using one f_node 
#$ -l h_rt=0:10:00    # Execution time
#$ -N serial          
. /etc/profile.d/modules.sh # Initialize module command
module load intel
module load mpt/2.16
module load fftw/3.3.8

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=true

./vlp4d.bdw_omp SLD10.dat
