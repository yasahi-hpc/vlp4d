# About

The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. Vlasov solver is typically based on a directional Strang splitting. The Poisson equation is treated with 2D Fourier transforms. For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.

The Vlasov solver is based on advection's operators: 
- 1D advection along x (Dt/2)
- 1D advection along y (Dt/2)
- Poisson solver -> compute electric fields Ex and E
- 1D advection along vx (Dt)
- 1D advection along vy (Dt)
- 1D advection along x (Dt/2)
- 1D advection along y (Dt/2)

Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default). 

Detailed descriptions of the test cases can be found in 
- Crouseilles & al. J. Comput. Phys., 228, pp. 1429-1446, (2009). 
  Section 5.3.1 Two-dimensional Landau damping -> SLD10
  http://people.rennes.inria.fr/Nicolas.Crouseilles/loss4D.pdf
- Crouseilles & al. Communications in Nonlinear Science and Numerical Simulation, pp 94-99, 13, (2008).
  Section 2 and 3 Two stream Instability and Beam focusing pb -> TSI20
  http://people.rennes.inria.fr/Nicolas.Crouseilles/cgls2.pdf
- Crouseilles & al. Beam Dynamics Newsletter no 41 (2006).
  Section 3.3, Beam focusing pb.
  http://icfa-bd.kek.jp/Newsletter41.pdf

# HPC
From the view point of high perfomrance computing (HPC), the code is parallelized with OpenMP without MPI domain decomposition.
In order to investigate the performance portability of this kind of kinietic plasma simulation codes, we implement the mini-app with
a mixed OpenACC/OpenMP and Kokkos, where we suppress unnecessary duplications of code lines. The detailed description is found in
- Yuuichi Asahi, Guillaume Latu, Virginie Grandgirard, and Julien Bigot, (Performance Portable Implementation of a Kinetic Plasma Simulation Mini-app, in Proceedings of Sixth Workshop on Accelerator Programming Using Directives (WACCPD), IEEE, 2019. (accepted)
https://sc19.supercomputing.org/proceedings/workshops/workshop_files/ws_waccpd104s2-file1.pdf

# Test environments
We have tested the code on the following environments. 
- Nvidia Tesla p100 on Tsubame3.0 (Tokyo Tech, Japan)  
Compilers (cuda/8.0.61, pgi19.1)

- Nvidia Tesla v100 on Summit (OLCF, US)  
Compilers (cuda/10.1.168, pgi19.1)

- Intel Skylake on JFRS-1 (IFERC-CSC, Japan)  
Compilers (intel19.0.0.117)

- Marvell Thunder X2 on CEA Computing Complex (CEA, France)  
Compilers (armclang19.2.0)

# Usage
## Compile
Depending on your configuration, you may have to modify the Makefile.
You may add your configuration in the same way as 
```
ifneq (,$(findstring p100,$(DEVICES)))
CXXFLAGS=-O3 -I/apps/t3/sles12sp2/cuda/8.0.61/include -ta=nvidia:cc60 -Minfo -std=c++11 -DOWN_INDEX_SEQUENCE -DNO_ASSERT_IN_CONSTEXPR -DENABLE_OPENACC
CXX=pgc++
LDFLAGS = -Mcudalib=cufft -ta=nvidia:cc60 -acc
TARGET = vlp4d.p100_acc
endif
```


### OpenACC version
```
export DEVICE=device_name # choose the device_name from "p100", "v100", "bdw", "skx", "tx2"
cd src_openacc
make
```

### Kokkos version
```
export KOKKOS_PATH=your_kokkos_path # set your_kokkos_path
export DEVICE=device_name # choose the device_name from "p100", "v100", "bdw", "skx", "tx2"
export POLICY=3D # optional, in case using MDRangePolicy3D for the better performance
cd src_kokkos
make
```

## Run
Depending on your configuration, you may have to modify the job.sh in wk and sub.sh in wk/batch_scripts.

```
cd wk
./job.sh
gnuplot -e 'plot "nrj.out" u 2 w l, "nrj_SLD10" u 2; pause -1' 
```

To checkout if results are OK, the nrj curve should be close enough to nrj_SLD10.
