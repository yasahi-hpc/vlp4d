#!/bin/bash
# Begin LSF Directives
#BSUB -P CSC367             # Project name
#BSUB -W 2:00               # Requested maximum walltime [hh:]mm
#BSUB -nnodes 1             # Number of nodes
#BSUB -J vlp4d
#BSUB -o vlp4d.%J
#BSUB -e vlp4d.%J
module load xl
module load fftw

date
cp ./vlp4d.v100_omp4.5 $MEMBERWORK/csc367
cp SLD10.dat $MEMBERWORK/csc367
cd $MEMBERWORK/csc367

jsrun -n 1 -g1 --smpiargs="-disable_gpu_hooks" ./vlp4d.v100_omp4.5 SLD10.dat
cp nrj.out $LS_SUBCWD
