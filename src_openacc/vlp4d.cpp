/*
 * @brief The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
 *        From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
 *        Vlasov solver is typically based on a directional Strang splitting. 
 *        The Poisson equation is treated with 2D Fourier transforms. 
 *        For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.
 *        The Vlasov solver is based on advection's operators:
 *
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *        Poisson solver -> compute electric fields Ex and E
 *        1D advection along vx (Dt)
 *        1D advection along vy (Dt)
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *
 *        Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default).
 *
 *  @author
 *  @url    https://gitlab.maisondelasimulation.fr/GyselaX/vlp4d/tree/master
 *
 *  @date 31/July/2019
 *        Merged version of vlp4d code
 *        Can be compiled with OpenACC and OpenMP
 *
 *        total time: 0.155460 s (P100)
 *        total time: 4.512256 s (BDW)
 *        total time: 1.807085 s (BDW, original)
 */

#include "types.h"
#include "config.h"
#include "efield.h"
#include "field.h"
#include "diags.h"
#include "timestep.h"
#include "init.h"
#include "debug.h"
#include <cstdio>
#include <chrono>

int main (int narg, char* arg[]) {
  Config conf;
  view_4d fn, fnp1;
  Efield *ef = NULL;
  Diags *dg = NULL;

  #if ! defined( ENABLE_OPENACC )
    #pragma omp parallel default(shared)
    {
  #endif
      // Initialization
      if(narg == 2) {
       // * A file is given in parameter *
       #if ! defined( ENABLE_OPENACC )
         #pragma omp master
         {
       #endif
           printf("reading input file %s\n", arg[1]);
           fflush(stdout);
       #if ! defined( ENABLE_OPENACC )
         }
       #endif
       init(arg[1], &conf, fn, fnp1, &ef, &dg);
      }
      else {
       #if ! defined( ENABLE_OPENACC )
         #pragma omp master
         {
       #endif
           printf("argc != 2, reading 'data.dat' by default\n");
           fflush(stdout);
       #if ! defined( ENABLE_OPENACC )
         }
       #endif
       init("data.dat", &conf, fn, fnp1, &ef, &dg);
      }

      #if ! defined( ENABLE_OPENACC )
        #pragma omp barrier
      #endif
      int iter = 0;

      // Declare timers
      std::chrono::high_resolution_clock::time_point t1, t2;
      t1 = std::chrono::high_resolution_clock::now();

      field_rho(&conf, fn, ef);
      field_poisson(&conf, ef, dg, iter);

      while(iter < conf.dom_.nbiter_) {
        #if ! defined( ENABLE_OPENACC )
          #pragma omp master
        #endif
        printf("iter %d\n", iter);

        iter++;
        onetimestep(&conf, fn, fnp1, ef, dg, iter);
      }

      t2 = std::chrono::high_resolution_clock::now();

      double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

      #if ! defined( ENABLE_OPENACC )
        #pragma omp master
      #endif
      printf("total time: %f s\n", seconds);

      finalize(&conf, fn, fnp1, &ef, &dg);

  #if ! defined( ENABLE_OPENACC )
    }
  #endif
  return 0;
}
