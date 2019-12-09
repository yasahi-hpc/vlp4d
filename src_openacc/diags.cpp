#include "diags.h"
#include "index.h"
#if ! defined( ENABLE_OPENACC )
  #include <omp.h>
#else
  #include <openacc.h>
#endif

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_+1;

  allocate(nrj_, {size_t(nbiter)});
  allocate(mass_, {size_t(nbiter)});
}

Diags::~Diags() {
  deallocate(nrj_);
  deallocate(mass_);
}

void Diags::compute(Config* conf, Efield* ef, int iter) {
  const Domain* dom = &conf->dom_;
  float64 iter_mass = 0.;
  float64 iter_nrj = 0.;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);

  float64 *dptr_rho = ef->rho_.raw();
  float64 *dptr_ex  = ef->ex_.raw();
  float64 *dptr_ey  = ef->ey_.raw();
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_rho,dptr_ex,dptr_ey)
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc kernels loop gang
    #else
      #pragma omp parallel for
    #endif
      for(int iy = 0; iy < ny; iy++) {
        #if defined( ENABLE_OPENACC )
          #pragma acc loop vector reduction (+:iter_mass,iter_nrj)
        #endif
        for(int ix = 0; ix < nx; ix++) {
          int idx = Index::coord_2D2int(ix,iy,nx,ny);
          const float64 eex = dptr_ex[idx];
          const float64 eey = dptr_ey[idx];
          const float64 rho = dptr_rho[idx];

          #if ! defined( ENABLE_OPENACC )
            #pragma omp atomic
          #endif
          iter_mass += rho;

          #if ! defined( ENABLE_OPENACC )
            #pragma omp atomic
          #endif
          iter_nrj += eex * eex + eey * eey;
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #endif

  iter_nrj = sqrt(iter_nrj * dom->dx_[0] * dom->dx_[1]);
  iter_mass *= dom->dx_[0] * dom->dx_[1];

  iter_nrj = iter_nrj > 1.e-30 ? log(iter_nrj) : -1.e9;

  nrj_[iter] = iter_nrj;
  mass_[iter] = iter_mass;
}

void Diags::save(Config* conf) {
  const Domain* dom = &conf->dom_;

  char filename[16];
  sprintf(filename, "nrj.out");

  FILE* fileid = fopen(filename, "w");
  for(int iter = 0; iter <= dom->nbiter_; ++iter)
      fprintf(fileid, "%17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_[iter], mass_[iter]);
  fclose(fileid);
}
