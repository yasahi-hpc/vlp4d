#include "diags.h"
#include "index.h"
#include "helper.h"
#include <stdio.h>

struct MomentFunctor {
  Config* conf_;  
  Efield* ef_;
  int s_nxmax, s_nymax;

  MomentFunctor(Config *conf, Efield *ef)
    : conf_(conf), ef_(ef) {
    const Domain *dom = &(conf_->dom_);
    s_nxmax  = dom->nxmax_[0];
    s_nymax  = dom->nxmax_[1];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const int &ixy, double_pair &lsum) const {
    int2 idx_2D = Index::int2coord_2D(ixy, s_nxmax, s_nymax);
    int ix = idx_2D.x, iy = idx_2D.y;
    const double eex = ef_->ex_(ix, iy);
    const double eey = ef_->ey_(ix, iy);

    lsum.x += ef_->rho_(ix, iy);
    lsum.y += eex * eex + eey * eey;
  }
};

Diags::Diags(Config *conf)
{
  const int nbiter = conf->dom_.nbiter_ + 1;

  nrj_ = view_1d_host("nrj", nbiter);
  mass_ = view_1d_host("mass", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter)
{
  const Domain *dom = &conf->dom_;
  double iter_mass = 0.;
  double iter_nrj = 0.;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  view_2d ex  = ef->ex_; 
  view_2d ey  = ef->ey_; 
  view_2d rho = ef->rho_; 

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  double_pair sum; sum.x = 0; sum.y = 0;
  Kokkos::parallel_reduce(nx * ny, KOKKOS_LAMBDA (const int ixy, double_pair& lsum) {
    int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;
    const double eex = ex(ix, iy);
    const double eey = ey(ix, iy);

    lsum.x += rho(ix, iy);
    lsum.y += eex * eex + eey * eey;
  }, sum);

  iter_mass = sum.x;
  iter_nrj = sum.y;

  iter_nrj = sqrt(iter_nrj * dom->dx_[0] * dom->dx_[1]);
  iter_mass *= dom->dx_[0] + dom->dx_[1];

  iter_nrj = iter_nrj > 1.e-30 ? log(iter_nrj) : -1.e9;
  nrj_(iter) = iter_nrj;
  mass_(iter) = iter_mass;
}

void Diags::save(Config *conf)
{
  const Domain* dom = &conf->dom_;

  char filename[16];
  sprintf(filename, "nrj.out");

  FILE *fileid = fopen(filename, "w");
  for(int iter=0; iter<=dom->nbiter_; ++iter)
    fprintf(fileid, "%17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_(iter), mass_(iter));

  fclose(fileid);
}

Diags::~Diags() {};
