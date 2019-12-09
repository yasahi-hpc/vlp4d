#include "config.h"
#include "advect.h"
#include "index.h"
#if ! defined( ENABLE_OPENACC )
  #include <omp.h>
#else
  #include <openacc.h>
  #include <accelmath.h>
#endif

#ifndef LAG_ORDER
  #define LAG_ORDER 5
#endif

#if LAG_ORDER == 4
  #define LAG_OFFSET 2
  #define LAG_HALO_PTS 3
  #define LAG_PTS 5

  static inline void lag_basis(double posx, double coef[LAG_PTS]) {
    const double loc[] = { 1. / 24., -1. / 6., 1. / 4., -1. / 6., 1. / 24. };
    coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
    coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
    coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
  }
#endif

#if LAG_ORDER == 5
  #define LAG_OFFSET 2
  #define LAG_HALO_PTS 3
  #define LAG_PTS 6
  #define LAG_ODD
  
  static inline void lag_basis(double px, double coef[LAG_PTS]) {
    const double loc[] = { -1. / 24, 1. / 24., -1. / 12., 1. / 12., -1. / 24., 1. / 24. };
    const double pxm2 = px - 2.;
    const double sqrpxm2 = pxm2 * pxm2;
    const double pxm2_01 = pxm2 * (pxm2 - 1.);
    coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
    coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
    coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) * (5 * sqrpxm2 - 3 * pxm2 - 6.);
    coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) * (5 * sqrpxm2 - 7 * pxm2 - 4.);
    coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
    coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
  }
#endif

#if LAG_ORDER == 7
  #define LAG_OFFSET 3
  #define LAG_HALO_PTS 4
  #define LAG_PTS 8
  #define LAG_ODD
  
  static inline void lag_basis(double px, double coef[LAG_PTS]) {
    const double loc[] = {-1. / 720, 1. / 720, -1. / 240, 1. / 144, -1. / 144, 1. / 240, -1. / 720, 1. / 720};
    const double pxm3 = px - 3.;
    const double sevenpxm3sqr = 7. * pxm3 * pxm3;
    const double f1t3 = (px - 1.) * (px - 2.) * (px - 3.);
    const double f4t6 = (px - 4.) * (px - 5.) * (px - 6.);
    coef[0] = loc[0] * f1t3 * f4t6 * (px - 4.);
    coef[1] = loc[1] * (px - 2.) * pxm3 * f4t6 * (sevenpxm3sqr + 8. * pxm3 - 18.);
    coef[2] = loc[2] * (px - 1.) * pxm3 * f4t6 * (sevenpxm3sqr + 2. * pxm3 - 15.);
    coef[3] = loc[3] * (px - 1.) * (px - 2.) * f4t6 * (sevenpxm3sqr - 4. * pxm3 - 12.);
    coef[4] = loc[4] * f1t3 * (px - 5.) * (px - 6.) * (sevenpxm3sqr - 10. * pxm3 - 9.);
    coef[5] = loc[5] * f1t3 * (px - 4.) * (px - 6.) * (sevenpxm3sqr - 16. * pxm3 - 6.);
    coef[6] = loc[6] * f1t3 * (px - 4.) * (px - 5.) * (sevenpxm3sqr - 22. * pxm3 - 3.);
    coef[7] = loc[7] * f1t3 * f4t6 * pxm3;
  }
#endif

void advect_1D_x(Config* conf, const view_4d fn, view_4d fnp1, double dt) {
  const Domain* dom = &conf->dom_;

  const int s_nxmax  = dom->nxmax_[0];
  const int s_nymax  = dom->nxmax_[1];
  const int s_nvxmax = dom->nxmax_[2];
  const int s_nvymax = dom->nxmax_[3];

  const double hmin     = dom->minPhy_[0];
  const double hwidth   = dom->maxPhy_[0] - dom->minPhy_[0];
  const double dh       = dom->dx_[0];
  const double idh      = 1. / dom->dx_[0];
  const double minPhyvx = dom->minPhy_[2];
  const double dvx      = dom->dx_[2];

  float64 *dptr_fn   = fn.raw();
  float64 *dptr_fnp1 = fnp1.raw();

  #if defined( ENABLE_OPENACC )
    const int n = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
  #endif

  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_fn[0:n],dptr_fnp1[0:n])
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(3)
    #else
      #pragma omp for schedule(static) collapse(2)
    #endif
      for(int ivy = 0; ivy < s_nvymax; ivy++) {
        for(int ivx = 0; ivx < s_nvxmax; ivx++) {
          for(int iy = 0; iy < s_nymax; iy++) {
            #if defined( ENABLE_OPENACC )
              #pragma acc loop vector independent
            #endif
            for(int ix = 0; ix < s_nxmax; ix++) {
              const double vx = minPhyvx + ivx * dvx;
              const double depx = dt * vx;
              const double x = hmin + ix * dh;
              const double xstar = hmin + fmod(hwidth + x - depx - hmin, hwidth);
              double coef[LAG_PTS];
              double ftmp = 0.;

              #ifdef LAG_ODD
                int ipos1 = floor((xstar - hmin) * idh);
              #else
                int ipos1 = round((xstar - hmin) * idh);
              #endif

              const double d_prev1 = LAG_OFFSET + idh * (xstar - (hmin + ipos1 * dh));
              ipos1 -= LAG_OFFSET;
              lag_basis(d_prev1, coef);

              #if defined( ENABLE_OPENACC )
                #pragma acc loop seq
              #endif
              for(int k=0; k<=LAG_ORDER; k++) {
                int idx_ipos1 = (s_nxmax + ipos1 + k) % s_nxmax;
                int idx = Index::coord_4D2int(idx_ipos1,iy,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
                ftmp += coef[k] * dptr_fn[idx];
              }
              int idx = Index::coord_4D2int(ix,iy,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
              dptr_fnp1[idx] = ftmp;
            }
          }
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #endif
}

void print_fxvx(Config* conf, view_4d fn, int iter) {
  Domain* dom = &(conf->dom_);
  char filename[128];
  printf("print_fxvx %d\n", iter);
  sprintf(filename, "fxvx%04d.out", iter);
  FILE* fileid = fopen(filename, "w");

  const size_t ivy = dom->nxmax_[3] / 2;
  const size_t iy = 0;

  for(size_t ivx = 0; ivx < dom->nxmax_[2]; ivx++) {
    for(size_t ix = 0; ix < dom->nxmax_[0]; ix++)
      fprintf(fileid, "%4lu %4lu %20.13le\n", ix, ivx, fn[ivy][ivx][iy][ix]);
    fprintf(fileid, "\n");
  }

  fclose(fileid);
}

void advect_1D_y(Config* conf, const view_4d fn, view_4d fnp1, double dt) {
  const Domain* dom = &conf->dom_;

  const int s_nxmax  = dom->nxmax_[0];
  const int s_nymax  = dom->nxmax_[1];
  const int s_nvxmax = dom->nxmax_[2];
  const int s_nvymax = dom->nxmax_[3];

  const double hmin     = dom->minPhy_[1];
  const double hwidth   = dom->maxPhy_[1] - dom->minPhy_[1];
  const double dh       = dom->dx_[1];
  const double idh      = 1. / dom->dx_[1];
  const double minPhyvy = dom->minPhy_[3];
  const double dvy      = dom->dx_[3];
  float64 *dptr_fn   = fn.raw();
  float64 *dptr_fnp1 = fnp1.raw();

  #if defined( ENABLE_OPENACC )
    const int n = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
  #endif

  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_fn[0:n],dptr_fnp1[0:n])
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(3)
    #else
      #pragma omp for schedule(static) collapse(2)
    #endif
      for(int ivy = 0; ivy < s_nvymax; ivy++) {
        for(int ivx = 0; ivx < s_nvxmax; ivx++) {
          for(int iy = 0; iy < s_nymax; iy++) {
            #if defined( ENABLE_OPENACC )
              #pragma acc loop vector independent
            #endif
            for(int ix = 0; ix < s_nxmax; ix++) {
              const double vy = minPhyvy + ivy * dvy;
              const double deph = dt * vy;
              const double y = hmin + iy * dh;
              const double hstar = hmin + fmod(hwidth + y - deph - hmin, hwidth);
              double coef[LAG_PTS];
              double ftmp = 0.;

              #ifdef LAG_ODD
                int ipos1 = floor((hstar - hmin) * idh);
              #else
                int ipos1 = round((hstar - hmin) * idh);
              #endif

              const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 * dh));
              ipos1 -= LAG_OFFSET;
              lag_basis(d_prev1, coef);

              #if defined( ENABLE_OPENACC )
                #pragma acc loop seq
              #endif
              for(int k=0; k<=LAG_ORDER; k++) {
                int idx_ipos1 = (s_nymax + ipos1 + k) % s_nymax;
                int idx = Index::coord_4D2int(ix,idx_ipos1,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
                ftmp += coef[k] * dptr_fn[idx];
              }
              int idx = Index::coord_4D2int(ix,iy,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
              dptr_fnp1[idx] = ftmp;
            }
          }
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #endif
}

void debug_1D_y(Config* conf, cview_4d fn) {
  Domain* dom = &(conf->dom_);
  static int count = 0;
  char filename[32];

  sprintf(filename, "fyvy%04d.out", count++);
  FILE* fileid = fopen(filename, "w");

  const size_t ivx = 0;
  const size_t ix = 0;

  for(size_t ivy = 0; ivy < dom->nxmax_[2]; ivy++) {
    for(size_t iy = 0; iy < dom->nxmax_[0]; iy++)
      fprintf(fileid, "%4lu %4lu %20.13le\n", ivy, iy, fn[ivy][ivx][iy][ix]);

    fprintf(fileid, "\n");
  }

  fclose(fileid);
}

void advect_1D_vx(Config* conf, const view_4d fn, view_4d fnp1, Efield* ef, double dt) {
  const Domain* dom = &conf->dom_;

  const int s_nxmax  = dom->nxmax_[0];
  const int s_nymax  = dom->nxmax_[1];
  const int s_nvxmax = dom->nxmax_[2];
  const int s_nvymax = dom->nxmax_[3];
  const double hmin   = dom->minPhy_[2];
  const double hwidth = dom->maxPhy_[2] - dom->minPhy_[2];
  const double dh     = dom->dx_[2];
  const double idh    = 1. / dom->dx_[2];

  float64 *dptr_fn   = fn.raw();
  float64 *dptr_fnp1 = fnp1.raw();
  float64 *dptr_ex   = ef->ex_.raw();

  #if defined( ENABLE_OPENACC )
    const int n1 = s_nxmax * s_nymax * s_nvxmax * s_nvymax, n2 = s_nxmax * s_nymax;
  #endif
  
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_fn[0:n1],dptr_fnp1[0:n1],dptr_ex[0:n2])
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(3)
    #else
      #pragma omp for schedule(static) collapse(2)
    #endif
      for(int ivy = 0; ivy < s_nvymax; ivy++) {
        for(int ivx = 0; ivx < s_nvxmax; ivx++) {
          for(int iy = 0; iy < s_nymax; iy++) {
            #if defined( ENABLE_OPENACC )
              #pragma acc loop vector independent
            #endif
            for(int ix = 0; ix < s_nxmax; ix++) {
              const double vx = hmin + ivx * dh;
              const double deph = dt * (dptr_ex[Index::coord_2D2int(ix,iy,s_nxmax,s_nymax)]);
              const double hstar = hmin + fmod(hwidth + vx - deph - hmin, hwidth);
              double coef[LAG_PTS];
              double ftmp = 0.;

              #ifdef LAG_ODD
                int ipos1 = floor((hstar - hmin) * idh);
              #else
                int ipos1 = round((hstar - hmin) * idh);
              #endif

              const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 * dh));
              ipos1 -= LAG_OFFSET;
              lag_basis(d_prev1, coef);

              #if defined( ENABLE_OPENACC )
                #pragma acc loop seq
              #endif
              for(int k = 0; k <= LAG_ORDER; k++) {
                int idx_ipos1 = (s_nvxmax + ipos1 + k) % s_nvxmax;
                int idx = Index::coord_4D2int(ix,iy,idx_ipos1,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
                ftmp += coef[k] * dptr_fn[idx];
              }
                      
              int idx = Index::coord_4D2int(ix,iy,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
              dptr_fnp1[idx] = ftmp;
            }
          }
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #endif
}

void advect_1D_vy(Config* conf, const view_4d fn, view_4d fnp1, Efield* ef, double dt) {
  const Domain* dom = &conf->dom_;

  const int s_nxmax   = dom->nxmax_[0];
  const int s_nymax   = dom->nxmax_[1];
  const int s_nvxmax  = dom->nxmax_[2];
  const int s_nvymax  = dom->nxmax_[3];
  const double hmin   = dom->minPhy_[3];
  const double hwidth = dom->maxPhy_[3] - dom->minPhy_[3];
  const double dh     = dom->dx_[3];
  const double idh    = 1. / dom->dx_[3];

  float64 *dptr_fn   = fn.raw();
  float64 *dptr_fnp1 = fnp1.raw();
  float64 *dptr_ey   = ef->ey_.raw();

  #if defined( ENABLE_OPENACC )
    const int n1 = s_nxmax * s_nymax * s_nvxmax * s_nvymax, n2 = s_nxmax * s_nymax;
  #endif

  #if defined( ENABLE_OPENACC )
    #pragma acc data present(dptr_fn[0:n1],dptr_fnp1[0:n1],dptr_ey[0:n2])
    {
  #endif
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(3)
    #else
      #pragma omp for schedule(static) collapse(2)
    #endif
      for(int ivy = 0; ivy < s_nvymax; ivy++) {
        for(int ivx = 0; ivx < s_nvxmax; ivx++) {
          for(int iy = 0; iy < s_nymax; iy++) {
            #if defined( ENABLE_OPENACC )
              #pragma acc loop vector independent
            #endif
            for(int ix = 0; ix < s_nxmax; ix++) {
              const double vy = hmin + ivy * dh;
              const double deph = dt * (dptr_ey[Index::coord_2D2int(ix,iy,s_nxmax,s_nymax)]);
              const double hstar = hmin + fmod(hwidth + vy - deph - hmin, hwidth);
              double coef[LAG_PTS];
              double ftmp = 0.;

              #ifdef LAG_ODD
                int ipos1 = floor((hstar - hmin) * idh);
              #else
                int ipos1 = round((hstar - hmin) * idh);
              #endif

              const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 * dh));
              ipos1 -= LAG_OFFSET;
              lag_basis(d_prev1, coef);

              #if defined( ENABLE_OPENACC )
                #pragma acc loop seq
              #endif
              for(int k = 0; k <= LAG_ORDER; k++) {
                int idx_ipos1 = (s_nvymax + ipos1 + k) % s_nvymax;
                int idx = Index::coord_4D2int(ix,iy,ivx,idx_ipos1,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
                ftmp += coef[k] * dptr_fn[idx];
              }
                      
              int idx = Index::coord_4D2int(ix,iy,ivx,ivy,s_nxmax,s_nymax,s_nvxmax,s_nvymax);
              dptr_fnp1[idx] = ftmp;
            }
          }
        }
      }
  #if defined( ENABLE_OPENACC )
    }
  #endif
}
