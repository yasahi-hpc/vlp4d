#include "efield.h"

// field init
Efield::Efield(Config *conf, shape_t<2> dim)
{
  rho_ = view_2d("rho", dim[0], dim[1]);
  ex_  = view_2d("ex",  dim[0], dim[1]);
  ey_  = view_2d("ey",  dim[0], dim[1]);
  phi_ = view_2d("phi", dim[0], dim[1]);

  // Initialize fft helper
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0];
  int ny = dom->nxmax_[1];
  double xmax = dom->maxPhy_[0];
  double kx0 = 2 * M_PI / xmax;

  fft_ = new Impl::FFT(nx, ny, 1);
  int nx1h = fft_->nx1h_;
  rho_hat_ = complex_view_2d("rho_hat", nx1h, ny);
  ex_hat_  = complex_view_2d("ex_hat",  nx1h, ny);
  ey_hat_  = complex_view_2d("ey_hat",  nx1h, ny);

  filter_ = view_1d("filter", nx1h);

  // Initialize filter (0,1/k2,1/k2,1/k2,...)
  typename view_1d::HostMirror h_filter = Kokkos::create_mirror_view(filter_);
  h_filter(0) = 0.;
  for(int ix = 1; ix < nx1h; ix++) {
    double kx = ix * kx0;
    double k2 = kx * kx;
    h_filter(ix) = 1./k2;
  }
  Kokkos::deep_copy(filter_, h_filter);
}

Efield::~Efield()
{
  if(fft_ != NULL) delete fft_;
}

void Efield::solve_poisson_fftw(double xmax, double ymax)
{
  double kx0 = 2 * M_PI / xmax;
  double ky0 = 2 * M_PI / ymax;
  int nx1  = fft_->nx1_;
  int nx1h = fft_->nx1h_;
  int nx2  = fft_->nx2_;
  int nx2h = fft_->nx2h_;
  double normcoeff = 1. / (nx1 * nx2);
  const complex64 I = complex64(0., 1.);
  
  // Forward 2D FFT (Real to Complex)
  fft_->fft2(rho_.data(), rho_hat_.data());

  // Solve Poisson equation in Fourier space
  complex_view_2d ex_hat  = ex_hat_;
  complex_view_2d ey_hat  = ey_hat_;
  complex_view_2d rho_hat = rho_hat_;

  // In order to avoid zero division in vectorized way
  // filter[0] == 0, and filter[0:] == 1./(ix1*kx0)**2
  view_1d filter = filter_; 
   
  Kokkos::parallel_for(nx1h, KOKKOS_LAMBDA (const int ix1) {
    double kx = ix1 * kx0;
    {
      int ix2 = 0;
      double kx = ix1 * kx0;
      ex_hat(ix1, ix2) = -kx * I * rho_hat(ix1, ix2) * filter(ix1) * normcoeff;
      ey_hat(ix1, ix2) = 0.;
      rho_hat(ix1, ix2) = rho_hat(ix1, ix2) * filter(ix1) * normcoeff;
    }

    for(int ix2=1; ix2<nx2h; ix2++) {
      double ky = ix2 * ky0;
      double k2 = kx * kx + ky * ky;

      ex_hat(ix1, ix2) = -(kx/k2) * I * rho_hat(ix1, ix2) * normcoeff;
      ey_hat(ix1, ix2) = -(ky/k2) * I * rho_hat(ix1, ix2) * normcoeff;
      rho_hat(ix1, ix2) = rho_hat(ix1, ix2) / k2 * normcoeff;
    }

    for(int ix2=nx2h; ix2<nx2; ix2++) {
      double ky = (ix2-nx2) * ky0;
      double k2 = kx*kx + ky*ky;

      ex_hat(ix1, ix2) = -(kx/k2) * I * rho_hat(ix1, ix2) * normcoeff;
      ey_hat(ix1, ix2) = -(ky/k2) * I * rho_hat(ix1, ix2) * normcoeff;
      rho_hat(ix1, ix2) = rho_hat(ix1, ix2) / k2 * normcoeff;
    }
  });

  // Backward 2D FFT (Complex to Real)
  fft_->ifft2(rho_hat.data(), rho_.data());
  fft_->ifft2(ex_hat.data(),  ex_.data());
  fft_->ifft2(ey_hat.data(),  ey_.data());
}
