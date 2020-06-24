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
 *  @date 24/June/2020
 *        OpenMP4.5 implementation of vlp4d code
 *
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

  // Initialization
  if(narg == 2) {
   // * A file is given in parameter *
   printf("reading input file %s\n", arg[1]);
   fflush(stdout);
   init(arg[1], &conf, fn, fnp1, &ef, &dg);
  }
  else {
   printf("argc != 2, reading 'donnees.dat' by default\n");
   fflush(stdout);
   init("donnees.dat", &conf, fn, fnp1, &ef, &dg);
  }

  int iter = 0;

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;
  t1 = std::chrono::high_resolution_clock::now();

  field_rho(&conf, fn, ef);
  field_poisson(&conf, ef, dg, iter);

  while(iter < conf.dom_.nbiter_) {
    std::cout << "iter " << iter << std::endl;

    iter++;
    onetimestep(&conf, fn, fnp1, ef, dg, iter);
  }

  t2 = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
  printf("total time: %f s\n", seconds);

  finalize(&conf, fn, fnp1, &ef, &dg);

  return 0;
}
