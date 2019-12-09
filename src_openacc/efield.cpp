#include "efield.h"
#include "index.h"
#if ! defined( ENABLE_OPENACC )
  #include <omp.h>
#else
  #include <openacc.h>
#endif

// field init
Efield::Efield(Config *conf, shape_t<2> dim) {
  allocate(rho_, dim);
  allocate(ex_,  dim);
  allocate(ey_,  dim);
  allocate(phi_, dim);

  // Initialize fft helper
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0];
  int ny = dom->nxmax_[1];
  float64 xmax = dom->maxPhy_[0];
  float64 kx0 = 2 * M_PI / xmax;

  fft_ = new Impl::FFT(nx, ny, 1);
  int nx1h = fft_->nx1h_;
  shape_t<2> dim_hat = {static_cast<size_t>(nx1h), static_cast<size_t>(ny)};
  allocate(rho_hat_, dim_hat);
  allocate(ex_hat_,  dim_hat);
  allocate(ey_hat_,  dim_hat);

  allocate(filter_, {static_cast<size_t>(nx1h)});

  // Initialize filter (0,1/k2,1/k2,1/k2,...)
  filter_[0] = 0.;
  for(int ix = 1; ix < nx1h; ix++) {
    float64 kx = ix * kx0;
    float64 k2 = kx * kx;
    filter_[ix] = 1./k2;
  }

  // Deep copy to GPUs 
  #if defined( ENABLE_OPENACC )
    int n1 = nx*ny, n2 = nx1h * ny;
    n1_ = n1; n2_ = n2; nx1h_ = nx1h;
    float64   *dptr_rho     = rho_.raw();
    float64   *dptr_ex      = ex_.raw();
    float64   *dptr_ey      = ey_.raw();
    float64   *dptr_phi     = phi_.raw();
    float64   *dptr_filter  = filter_.raw();
    complex64 *dptr_rho_hat = rho_hat_.raw();
    complex64 *dptr_ex_hat  = ex_hat_.raw();
    complex64 *dptr_ey_hat  = ey_hat_.raw();
    #pragma acc enter data copyin(this) // shallow copy
    #pragma acc enter data create(dptr_rho[0:n1],dptr_ex[0:n1],dptr_ey[0:n1],dptr_phi[0:n1]) // deep copy data members
    #pragma acc enter data create(dptr_rho_hat[0:n2],dptr_ex_hat[0:n2],dptr_ey_hat[0:n2],dptr_filter[0:nx1h]) // deep copy data members
    #pragma acc update device(dptr_filter[0:nx1h]) // update filter
  #endif
}

Efield::~Efield() {
  #if defined( ENABLE_OPENACC )
    int n1 = n1_, n2 = n2_, nx1h = nx1h_;
    float64   *dptr_rho     = rho_.raw();
    float64   *dptr_ex      = ex_.raw();
    float64   *dptr_ey      = ey_.raw();
    float64   *dptr_phi     = phi_.raw();
    float64   *dptr_filter  = filter_.raw();
    complex64 *dptr_rho_hat = rho_hat_.raw();
    complex64 *dptr_ex_hat  = ex_hat_.raw();
    complex64 *dptr_ey_hat  = ey_hat_.raw();
    #pragma acc exit data delete(dptr_rho[0:n1],dptr_ex[0:n1],dptr_ey[0:n1],dptr_phi[0:n1]) // remove data members
    #pragma acc exit data delete(dptr_rho_hat[0:n2],dptr_ex_hat[0:n2],dptr_ey_hat[0:n2],dptr_filter[0:nx1h]) // remove data members
    #pragma acc exit data delete(this) // delete this pointer
  #endif

  if(fft_ != NULL) delete fft_;

  deallocate(rho_);
  deallocate(ex_);
  deallocate(ey_);
  deallocate(phi_);

  deallocate(rho_hat_);
  deallocate(ex_hat_);
  deallocate(ey_hat_);
  deallocate(filter_);
}

void Efield::solve_poisson_fftw(float64 xmax, float64 ymax) {
  float64 kx0 = 2 * M_PI / xmax;
  float64 ky0 = 2 * M_PI / ymax;
  int nx1  = fft_->nx1_;
  int nx1h = fft_->nx1h_;
  int nx2  = fft_->nx2_;
  int nx2h = fft_->nx2h_;
  float64 normcoeff = 1. / (nx1 * nx2);
  const complex64 I = complex64(0., 1.);
  
  // Use local pointer to avoid issues with "use_device" [See Known limitations 3.5] https://www.pgroup.com/resources/docs/19.1/x86/openacc-gs/index.htm
  float64 *dptr_rho = rho_.raw(), *dptr_ex = ex_.raw(), *dptr_ey = ey_.raw(), *dptr_filter = filter_.raw();
  complex64 *dptr_rho_hat = rho_hat_.raw(), *dptr_ex_hat = ex_hat_.raw(), *dptr_ey_hat = ey_hat_.raw();
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_rho,dptr_ex,dptr_ey,dptr_rho_hat,dptr_ex_hat,dptr_ey_hat,dptr_filter)
    {
  #endif

    // Forward 2D FFT (Real to Complex)
    #if defined( ENABLE_OPENACC )
      #pragma acc host_data use_device(dptr_rho, dptr_rho_hat)
    #endif
    fft_->fft2(dptr_rho, dptr_rho_hat);

    // Solve Poisson equation in Fourier space
    // In order to avoid zero division in vectorized way
    // filter[0] == 0, and filter[0:] == 1./(ix1*kx0)**2
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop
      for(int ix1=0; ix1<nx1h; ix1++) {
        int ix2 = 0;
        int idx = Index::coord_2D2int(ix1,ix2,nx1h,nx2);
        float64 kx = ix1 * kx0;

        dptr_ex_hat[idx]  = -kx * I * dptr_rho_hat[idx] * dptr_filter[ix1] * normcoeff;
        dptr_ey_hat[idx]  = 0.;
        dptr_rho_hat[idx] = dptr_rho_hat[idx] * dptr_filter[ix1] * normcoeff;

        #pragma acc loop seq
        for(int ix2=1; ix2<nx2h; ix2++) {
          float64 kx = ix1 * kx0;
          float64 ky = ix2 * ky0;
          float64 k2 = kx * kx + ky * ky;
          int idx = Index::coord_2D2int(ix1,ix2,nx1h,nx2);
  
          dptr_ex_hat[idx]  = -(kx/k2) * I * dptr_rho_hat[idx] * normcoeff;
          dptr_ey_hat[idx]  = -(ky/k2) * I * dptr_rho_hat[idx] * normcoeff;
          dptr_rho_hat[idx] = dptr_rho_hat[idx] / k2 * normcoeff;
        }
  
        #pragma acc loop seq
        for(int ix2=nx2h; ix2<nx2; ix2++) {
          float64 kx = ix1 * kx0;
          float64 ky = (ix2-nx2) * ky0;
          float64 k2 = kx*kx + ky*ky;
          int idx = Index::coord_2D2int(ix1,ix2,nx1h,nx2);
  
          dptr_ex_hat[idx] = -(kx/k2) * I * dptr_rho_hat[idx] * normcoeff;
          dptr_ey_hat[idx] = -(ky/k2) * I * dptr_rho_hat[idx] * normcoeff;
          dptr_rho_hat[idx] = dptr_rho_hat[idx] / k2 * normcoeff;
        }
      }
    #else
      #pragma omp for schedule(static)
      for(int ix1=0; ix1<nx1h; ix1++) {
        int ix2 = 0;
        float64 kx = ix1 * kx0;
        ex_hat_[ix2][ix1] = -kx * I * rho_hat_[ix2][ix1] * filter_[ix1] * normcoeff;
        ey_hat_[ix2][ix1] = 0.;
        rho_hat_[ix2][ix1] = rho_hat_[ix2][ix1] * filter_[ix1] * normcoeff;
      }
      
      #pragma omp for schedule(static)
      for(int ix2=1; ix2<nx2h; ix2++) {
        for(int ix1=0; ix1<nx1h; ix1++) {
          float64 kx = ix1 * kx0;
          float64 ky = ix2 * ky0;
          float64 k2 = kx * kx + ky * ky;
        
          ex_hat_[ix2][ix1]  = -(kx/k2) * I * rho_hat_[ix2][ix1] * normcoeff;
          ey_hat_[ix2][ix1]  = -(ky/k2) * I * rho_hat_[ix2][ix1] * normcoeff;
          rho_hat_[ix2][ix1] = rho_hat_[ix2][ix1] / k2 * normcoeff;
        }
      }
      
      #pragma omp for schedule(static)
      for(int ix2=nx2h; ix2<nx2; ix2++) {
        for(int ix1=0; ix1<nx1h; ix1++) {
          float64 kx = ix1 * kx0;
          float64 ky = (ix2-nx2) * ky0;
          float64 k2 = kx*kx + ky*ky;
        
          ex_hat_[ix2][ix1] = -(kx/k2) * I * rho_hat_[ix2][ix1] * normcoeff;
          ey_hat_[ix2][ix1] = -(ky/k2) * I * rho_hat_[ix2][ix1] * normcoeff;
          rho_hat_[ix2][ix1] = rho_hat_[ix2][ix1] / k2 * normcoeff;
        }
      };
    #endif

    // Backward 2D FFT (Complex to Real)
    #if defined( ENABLE_OPENACC )
      #pragma acc host_data use_device(dptr_rho, dptr_rho_hat,dptr_ex,dptr_ex_hat,dptr_ey,dptr_ey_hat)
      {
    #endif
        fft_->ifft2(dptr_rho_hat, dptr_rho);
        fft_->ifft2(dptr_ex_hat,  dptr_ex);
        fft_->ifft2(dptr_ey_hat,  dptr_ey);
    #if defined( ENABLE_OPENACC )
      }
    #endif
  #if defined( ENABLE_OPENACC )
    }
  #endif
}
