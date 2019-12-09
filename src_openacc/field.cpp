#include "field.h"
#include "index.h"
#include "debug.h"
#if ! defined( ENABLE_OPENACC )
  #include <omp.h>
#else
  #include <openacc.h>
#endif

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, view_4d fn, Efield *ef)
{
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  float64 *dptr_fn  = fn.raw();
  float64 *dptr_rho = ef->rho_.raw();
  float64 *dptr_ex  = ef->ex_.raw();
  float64 *dptr_ey  = ef->ey_.raw();
  float64 *dptr_phi = ef->phi_.raw();
  #if defined( ENABLE_OPENACC )
    const int n1 = nx * ny, n2 = nx * ny * nvx * nvy;
  #endif

  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_fn[0:n2],dptr_rho[0:n1],dptr_ex[0:n1],dptr_ey[0:n1],dptr_phi[0:n1])
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(2)
    #else
      #pragma omp for schedule(static)
    #endif
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          dptr_rho[idx] = 0.;
          dptr_ex[idx]  = 0.;
          dptr_ey[idx]  = 0.;
          dptr_phi[idx] = 0.;
        }
      }

    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(4)
    #else
      #pragma omp for schedule(static)
    #endif
      for(int ivy=0; ivy<nvy; ivy++) {
        for(int ivx=0; ivx<nvx; ivx++) {
          for(int iy=0; iy<ny; iy++) {
            for(int ix=0; ix<nx; ix++) {
              int idx_src = Index::coord_4D2int(ix,iy,ivx,ivy,nx,ny,nvx,nvy);
              int idx_dst = Index::coord_2D2int(ix,iy,nx,ny);
              #if defined( ENABLE_OPENACC )
                #pragma acc atomic update
              #else
                #pragma omp atomic
              #endif
              dptr_rho[idx_dst] += dptr_fn[idx_src];
            }
          }
        }
      }

    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(2)
    #else
      #pragma omp for schedule(static)
    #endif
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          dptr_rho[idx] *= dvx * dvy;
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #else
    #pragma omp barrier
  #endif
};

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dx = dom->dx_[0], dy = dom->dx_[1];
  float64 minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  float64 *dptr_ex = ef->ex_.raw();
  float64 *dptr_ey = ef->ey_.raw();
  float64 *dptr_rho = ef->rho_.raw();
  #if defined( ENABLE_OPENACC )
    const int n1 = nx * ny;
  #endif
  switch(dom->idcase_) {
    case 2:
      #if defined( ENABLE_OPENACC )
        #pragma acc data present(dptr_ex[0:n1],dptr_ey[0:n1])
        {
      #endif
        #if defined( ENABLE_OPENACC )
          #pragma acc parallel loop collapse(2)
        #else
          #pragma omp for
        #endif
          for(int iy=0; iy<ny; iy++) {
            for(int ix=0; ix<nx; ix++) {
              int idx = Index::coord_2D2int(ix,iy,nx,ny);
              dptr_ex[idx] = -(minPhyx + ix * dx);
              dptr_ey[idx] = 0.;
            }
          }
      #if defined( ENABLE_OPENACC )
        }
      #endif

      break;

    case 6:
      #if defined( ENABLE_OPENACC )
        #pragma acc data present(dptr_ex[0:n1],dptr_ey[0:n1])
        {
      #endif
        #if defined( ENABLE_OPENACC )
          #pragma acc parallel loop collapse(2)
        #else
          #pragma omp for
        #endif
          for(int iy=0; iy<ny; iy++) {
            for(int ix=0; ix<nx; ix++) {
              int idx = Index::coord_2D2int(ix,iy,nx,ny);
              dptr_ey[idx] = -(minPhyy + iy * dy);
              dptr_ex[idx] = 0.;
            }
          }
      #if defined( ENABLE_OPENACC )
        }
      #endif
      break;

    case 10:
    case 20:
      #if defined( ENABLE_OPENACC )
        #pragma acc data present(dptr_rho[0:n1])
        {
      #endif
        #if defined( ENABLE_OPENACC )
          #pragma acc parallel loop collapse(2)
        #else
          #pragma omp for
        #endif
          for(int iy=0; iy<ny; iy++) {
            for(int ix=0; ix<nx; ix++) {
              int idx = Index::coord_2D2int(ix,iy,nx,ny);
              dptr_rho[idx] -= 1.;
            }
          }
      #if defined( ENABLE_OPENACC )
        }
      #endif
        
      lu_solve_poisson(conf, ef, dg, iter);
      break;

    default:
      lu_solve_poisson(conf, ef, dg, iter);
      break;
  }
};

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
  dg->compute(conf, ef, iter);
};
