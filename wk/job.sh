#!/bin/sh
if ls *.bdw_kokkos > /dev/null 2>&1; then
  # Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
  qsub -g jh190065 batch_scripts/sub_bdw_kokkos.sh
elif ls *.bdw_omp > /dev/null 2>&1; then
  # Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
  qsub -g jh190065 batch_scripts/sub_bdw_omp.sh
elif ls *.p100_kokkos > /dev/null 2>&1; then
  # Nvidia TeslaP100 on Tsubame3.0 (Tokyo Tech, Japan)
  qsub -g jh190065 batch_scripts/sub_p100_kokkos.sh
elif ls *.p100_acc > /dev/null 2>&1; then
  # Nvidia TeslaP100 on Tsubame3.0 (Tokyo Tech, Japan)
  qsub -g jh190065 batch_scripts/sub_p100_acc.sh
elif ls *.v100_kokkos > /dev/null 2>&1; then
  # Nvidia TeslaV100 on Summit (OLCF, US)
  bsub batch_scripts/sub_v100_kokkos.sh
elif ls *.v100_acc > /dev/null 2>&1; then
  # Nvidia TeslaV100 on Summit (OLCF, US)
  bsub batch_scripts/sub_v100_acc.sh
elif ls *.v100_omp4.5 > /dev/null 2>&1; then
  # Nvidia TeslaV100 on Summit (OLCF, US)
  bsub batch_scripts/sub_v100_omp4.5.sh
elif ls *.skx_kokkos > /dev/null 2>&1; then
  # Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
  sbatch batch_scripts/sub_skx_kokkos.sh
elif ls *.skx_omp > /dev/null 2>&1; then
  # Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
  sbatch batch_scripts/sub_skx_omp.sh
elif ls *.tx2_kokkos > /dev/null 2>&1; then
  # Marvell Thunder X2 on CEA Computing Complex (CEA, France)
  sbatch batch_scripts/sub_tx2_kokkos.sh
elif ls *.tx2_omp > /dev/null 2>&1; then
  # Marvell Thunder X2 on CEA Computing Complex (CEA, France)
  sbatch batch_scripts/sub_tx2_omp.sh
else
  echo "No executable!"
fi
