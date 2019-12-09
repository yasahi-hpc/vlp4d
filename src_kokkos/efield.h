#ifndef __EFIELD_H__
#define __EFIELD_H__

#include "types.h"
#include "config.h"

#if ! defined( KOKKOS_ENABLE_CUDA )
  #include "fftw.h"
#else
  #include "fft.h"
#endif

struct Efield {
  view_2d rho_;
  view_2d ex_;
  view_2d ey_;
  view_2d phi_;

  view_1d filter_; // [YA added] In order to avoid conditional to keep (0, 0) component 0

  Impl::FFT *fft_;

  // a 2D complex buffer of size nx1h * nx2 (renamed)
private:
  complex_view_2d rho_hat_;
  complex_view_2d ex_hat_;
  complex_view_2d ey_hat_;

public:
  Efield(Config *conf, shape_t<2> dim);
  virtual ~Efield();

  void solve_poisson_fftw(double xmax, double ymax);
};

#endif
