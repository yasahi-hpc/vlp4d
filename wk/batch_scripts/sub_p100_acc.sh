#!/bin/sh
#$ -cwd               # job execution in the current directory   
#$ -l f_node=1        # Using one f_node 
#$ -l h_rt=0:30:00    # Execution time
#$ -N serial          
. /etc/profile.d/modules.sh # Initialize module command
module load intel
module load cuda/8.0.61

./vlp4d.p100_acc SLD10.dat
