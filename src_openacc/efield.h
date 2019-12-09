#ifndef _EFIELD_H
#define _EFIELD_H

#include "config.h"
#include "view_nd.h"

#if ! defined( ENABLE_OPENACC )
  #include "fftw.h"
#else
  #include "fft.h"
#endif

struct Efield {
  view_2d rho_;
  view_2d ex_;
  view_2d ey_;
  view_2d phi_;

  Impl::FFT *fft_;

private:
  view_1d filter_; // [YA Added] In order to avoid conditional to keep (0, 0) component 0

  // 2D complex buffers of size nx1h * nx2 (renamed)
  complex_view_2d rho_hat_;
  complex_view_2d ex_hat_;
  complex_view_2d ey_hat_;

  int n1_, n2_, nx1h_;
  
public:
  Efield(Config *conf, shape_t<2> dim);
  virtual ~Efield();
  
  void solve_poisson_fftw(float64 xmax, float64 ymax);
};

#endif
