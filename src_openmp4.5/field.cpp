#include "field.h"
#include "index.h"
#include "debug.h"
#include <omp.h>

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, view_4d fn, Efield *ef)
{
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];
  float64 *dptr_fn = fn.raw();
  float64 dvxy = dvx*dvy;

  float64 *dptr_rho = ef->rho_.raw();
  float64 *dptr_ex  = ef->ex_.raw();
  float64 *dptr_ey  = ef->ey_.raw();
  float64 *dptr_phi = ef->phi_.raw();
  #if defined( ENABLE_OPENMP_OFFLOAD )
    #pragma omp target teams distribute parallel for simd collapse(2)
  #else
    #pragma omp parallel for schedule(static)
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

  #if defined( ENABLE_OPENMP_OFFLOAD )
    #pragma omp target teams distribute parallel for simd collapse(2)
  #else
    #pragma omp parallel for collapse(2)
  #endif
  for(int iy=0; iy<ny; iy++) {
    for(int ix=0; ix<nx; ix++) {
      float64 sum = 0.;
      for(int ivy=0; ivy<nvy; ivy++) {
        for(int ivx=0; ivx<nvx; ivx++) {
          int idx_src = Index::coord_4D2int(ix,iy,ivx,ivy,nx,ny,nvx,nvy);
          sum += dptr_fn[idx_src];
        }
      }
      int idx_dst = Index::coord_2D2int(ix,iy,nx,ny);
      dptr_rho[idx_dst] = sum * dvxy;
    }
  }
};

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dx = dom->dx_[0], dy = dom->dx_[1];
  float64 minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  float64 *dptr_ex  = ef->ex_.raw();
  float64 *dptr_ey  = ef->ey_.raw();
  float64 *dptr_rho = ef->rho_.raw();
  switch(dom->idcase_) {
    case 2:
      #if defined( ENABLE_OPENMP_OFFLOAD )
        #pragma omp target teams distribute parallel for simd collapse(2)
      #else
        #pragma omp parallel for schedule(static)
      #endif
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          dptr_ex[idx] = -(minPhyx + ix * dx);
          dptr_ey[idx] = 0.;
        }
      }

      break;

    case 6:
      #if defined( ENABLE_OPENMP_OFFLOAD )
        #pragma omp target teams distribute parallel for simd collapse(2)
      #else
        #pragma omp parallel for schedule(static)
      #endif
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          dptr_ey[idx] = -(minPhyy + iy * dy);
          dptr_ex[idx] = 0.;
        }
      }
      break;

    case 10:
    case 20:
      #if defined( ENABLE_OPENMP_OFFLOAD )
        #pragma omp target teams distribute parallel for simd collapse(2)
      #else
        #pragma omp parallel for schedule(static)
      #endif
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          dptr_rho[idx] -= 1.;
        }
      }
        
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
