#include "field.h"
#include "index.h"
#include "helper.h"

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, view_4d fn, Efield *ef)
{
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dvx = dom->dx_[2], dvy = dom->dx_[3];

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  view_2d rho = ef->rho_; 
  view_2d ex  = ef->ex_; 
  view_2d ey  = ef->ey_; 
  view_2d phi = ef->phi_; 

  Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
    int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;
    rho(ix, iy) = 0.;
    ex(ix, iy) = 0.;
    ey(ix, iy) = 0.;
    phi(ix, iy) = 0.;
  });

  Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
    int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;

    double sum = 0.;
    for(int ivy=0; ivy<nvy; ivy++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        sum += fn(ix, iy, ivx, ivy);
      }
    }
    rho(ix, iy) = sum;
  });

  Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
    int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;
    rho(ix, iy) *= dvx * dvy;
  });
};

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dx = dom->dx_[0], dy = dom->dx_[1];
  double minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  view_2d rho = ef->rho_; 
  view_2d ex  = ef->ex_; 
  view_2d ey  = ef->ey_; 
  switch(dom->idcase_)
  {
    case 2:
        Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
          int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
          int ix = idx_2D.x, iy = idx_2D.y;
          ex(ix, iy) = -(minPhyx + ix * dx);
          ey(ix, iy) = 0.;
        });
        break;
    case 6:
        Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
          int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
          int ix = idx_2D.x, iy = idx_2D.y;
          ey(ix, iy) = -(minPhyy + iy * dy);
          ex(ix, iy) = 0.;
        });
        break;
    case 10:
    case 20:
        Kokkos::parallel_for(nx*ny, KOKKOS_LAMBDA (const int ixy) {
          int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
          int ix = idx_2D.x, iy = idx_2D.y;
          rho(ix, iy) -= 1.;
        });
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
