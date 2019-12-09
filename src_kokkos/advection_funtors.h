#ifndef __ADVECTION_FUNCTORS_H__
#define __ADVECTION_FUNCTORS_H__

#include <Kokkos_Core.hpp>
#include "config.h"
#include "efield.h"
#include "types.h"
#include "index.h"

namespace Advection {

#ifndef LAG_ORDER
#define LAG_ORDER 5
#endif

#if LAG_ORDER == 4
#define LAG_OFFSET 2
#define LAG_HALO_PTS 3
#define LAG_PTS 5

KOKKOS_INLINE_FUNCTION
void lag_basis(double posx, double * coef) {
  const double loc[] = {1./24., -1./6., 1./4., -1./6., 1./24.};

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

KOKKOS_INLINE_FUNCTION
void lag_basis(double px, double * coef) {
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
#define LAG_PTS 7
#define LAG_ODD

KOKKOS_INLINE_FUNCTION
void lag_basis(const double px, double * coef) {
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

  struct advect_1D_x_functor {
    Config* conf_;
    view_4d fn_, fnp1_;
    double dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    double hmin, hwidth, dh, idh;
    double minPhix, minPhivx, dx, dvx;

    advect_1D_x_functor(Config*conf, const view_4d fn, view_4d fnp1, double dt)
      : conf_(conf), fn_(fn), fnp1_(fnp1), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[0];
      hwidth = dom->maxPhy_[0] - dom->minPhy_[0];
      dh = dom->dx_[0];
      idh = 1. / dom->dx_[0];

      minPhix  = dom->minPhy_[0];
      minPhivx = dom->minPhy_[2];
      dx       = dom->dx_[0];
      dvx      = dom->dx_[2];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int &i ) const {
      int4 idx_4D = Index::int2coord_4D(i, s_nxmax, s_nymax, s_nvxmax, s_nvymax);
      int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;

      const double x  = minPhix + ix * dx;
      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * vx;
      const double xstar = hmin + fmod(hwidth + x - depx - hmin, hwidth);

      double coef[LAG_PTS];
      double ftmp = 0.;

#ifdef LAG_ODD
      int ipos1 = floor((xstar-hmin) * idh);
#else
      int ipos1 = round((xstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (xstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);

      for(int k=0; k<=LAG_ORDER; k++) {
        int idx_ipos1 = (s_nxmax + ipos1 + k) % s_nxmax;
        ftmp += coef[k] * fn_(idx_ipos1, iy, ivx, ivy);
      }

      fnp1_(ix, iy, ivx, ivy) = ftmp;
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx)  const {
      const double x  = minPhix + ix * dx;
      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * vx;
      const double xstar = hmin + fmod(hwidth + x - depx - hmin, hwidth);
    
      double coef[LAG_PTS];
    
#ifdef LAG_ODD
      int ipos1 = floor((xstar-hmin) * idh);
#else
      int ipos1 = round((xstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (xstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);
    
      KOKKOS_SIMD
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nxmax + ipos1 + k) % s_nxmax;
          ftmp += coef[k] * fn_(idx_ipos1, iy, ivx, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_y_functor {
    Config* conf_;
    view_4d fn_, fnp1_;
    double dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    double hmin, hwidth, dh, idh;
    double minPhiy, minPhivy, dy, dvy;

    advect_1D_y_functor(Config* conf, view_4d fn, view_4d fnp1, double dt)
      : conf_(conf), fn_(fn), fnp1_(fnp1), dt_(dt) {
      const Domain *dom = &(conf->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[1];
      hwidth = dom->maxPhy_[1] - dom->minPhy_[1];
      dh = dom->dx_[1];
      idh = 1. / dom->dx_[1];

      minPhiy  = dom->minPhy_[1];
      minPhivy = dom->minPhy_[3];
      dy       = dom->dx_[1];
      dvy      = dom->dx_[3];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int &i ) const {
      int4 idx_4D = Index::int2coord_4D(i, s_nxmax, s_nymax, s_nvxmax, s_nvymax);
      int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;

      const double y  = minPhiy + iy * dy;
      const double vy = minPhivy + ivy * dvy;
      const double deph = dt_ * vy;

      const double hstar = hmin + fmod(hwidth + y - deph - hmin, hwidth);

      double coef[LAG_PTS];
      double ftmp = 0.;

#ifdef LAG_ODD
        int ipos1 = floor((hstar-hmin) * idh);
#else
        int ipos1 = round((hstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);

      for(int k=0; k<=LAG_ORDER; k++) {
        int idx_ipos1 = (s_nymax + ipos1 + k) % s_nymax;
        ftmp += coef[k] * fn_(ix, idx_ipos1, ivx, ivy);
      }
      fnp1_(ix, iy, ivx, ivy) = ftmp;
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx) const {
      const double y  = minPhiy + iy * dy;
    
      KOKKOS_SIMD
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        const double vy = minPhivy + ivy * dvy;
        const double deph = dt_ * vy;
        
        const double hstar = hmin + fmod(hwidth + y - deph - hmin, hwidth);
        double coef[LAG_PTS];
         
#ifdef LAG_ODD
        int ipos1 = floor((hstar-hmin) * idh);
#else
        int ipos1 = round((hstar-hmin) * idh);
#endif
        const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
        ipos1 -= LAG_OFFSET;
        lag_basis(d_prev1, coef);
        
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nymax + ipos1 + k) % s_nymax;
          ftmp += coef[k] * fn_(ix, idx_ipos1, ivx, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_vx_functor {
    Config* conf_;
    view_4d fn_, fnp1_;
    Efield* ef_;

    view_2d ex;
    double dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    double hmin, hwidth, dh, idh;
    double minPhivx, dvx;

    advect_1D_vx_functor(Config* conf, view_4d fn, view_4d fnp1, Efield* ef, double dt)
      : conf_(conf), fn_(fn), fnp1_(fnp1), ef_(ef), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[2];
      hwidth = dom->maxPhy_[2] - dom->minPhy_[2];
      dh = dom->dx_[2];
      idh = 1. / dom->dx_[2];

      minPhivx = dom->minPhy_[2];
      dvx      = dom->dx_[2];

      ex = ef_->ex_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int &i ) const {
      int4 idx_4D = Index::int2coord_4D(i, s_nxmax, s_nymax, s_nvxmax, s_nvymax);
      int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;

      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * ex(ix, iy);
      const double hstar = hmin + fmod(hwidth + vx - depx - hmin, hwidth);

      double coef[LAG_PTS];
      double ftmp = 0.;

#ifdef LAG_ODD
      int ipos1 = floor((hstar-hmin) * idh);
#else
      int ipos1 = round((hstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);

      for(int k=0; k<=LAG_ORDER; k++) {
        int idx_ipos1 = (s_nvxmax + ipos1 + k) % s_nvxmax;
        ftmp += coef[k] * fn_(ix, iy, idx_ipos1, ivy);
      }

      fnp1_(ix, iy, ivx, ivy) = ftmp;
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx) const {
      const double vx = minPhivx + ivx * dvx;
      const double depx = dt_ * ex(ix, iy);
      const double hstar = hmin + fmod(hwidth + vx - depx - hmin, hwidth);
    
      double coef[LAG_PTS];
    
#ifdef LAG_ODD
      int ipos1 = floor((hstar-hmin) * idh);
#else
      int ipos1 = round((hstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);
    
      KOKKOS_SIMD
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nvxmax + ipos1 + k) % s_nvxmax;
          ftmp += coef[k] * fn_(ix, iy, idx_ipos1, ivy);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  struct advect_1D_vy_functor {
    Config* conf_;
    view_4d fn_, fnp1_;
    Efield* ef_;

    view_2d ey;
    double dt_;
    int s_nxmax, s_nymax, s_nvxmax, s_nvymax;
    double hmin, hwidth, dh, idh;
    double minPhivy, dvy;

    advect_1D_vy_functor(Config* conf, view_4d fn, view_4d fnp1, Efield* ef, double dt)
      : conf_(conf), fn_(fn), fnp1_(fnp1), ef_(ef), dt_(dt) {
      const Domain *dom = &(conf_->dom_);
      s_nxmax  = dom->nxmax_[0];
      s_nymax  = dom->nxmax_[1];
      s_nvxmax = dom->nxmax_[2];
      s_nvymax = dom->nxmax_[3];

      hmin = dom->minPhy_[3];
      hwidth = dom->maxPhy_[3] - dom->minPhy_[3];
      dh = dom->dx_[3];
      idh = 1. / dom->dx_[3];

      minPhivy = dom->minPhy_[3];
      dvy      = dom->dx_[3];

      ey = ef_->ey_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int &i ) const {
      int4 idx_4D = Index::int2coord_4D(i, s_nxmax, s_nymax, s_nvxmax, s_nvymax);
      int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;

      const double vy = minPhivy + ivy * dvy;
      const double deph = dt_ * ey(ix, iy);
      const double hstar = hmin + fmod(hwidth + vy - deph - hmin, hwidth);

      double coef[LAG_PTS];
      double ftmp = 0.;

#ifdef LAG_ODD
      int ipos1 = floor((hstar-hmin) * idh);
#else
      int ipos1 = round((hstar-hmin) * idh);
#endif
      const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
      ipos1 -= LAG_OFFSET;
      lag_basis(d_prev1, coef);

      for(int k=0; k<=LAG_ORDER; k++) {
        int idx_ipos1 = (s_nvymax + ipos1 + k) % s_nvymax;
        ftmp += coef[k] * fn_(ix, iy, ivx, idx_ipos1);
      }

      fnp1_(ix, iy, ivx, ivy) = ftmp;
    }

    // 3DRange policy over x, y, vx directions and sequential loop along vy direction
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx) const {
      for(int ivy=0; ivy<s_nvymax; ivy++) {
        const double vy = minPhivy + ivy * dvy;
        const double deph = dt_ * ey(ix, iy);
        const double hstar = hmin + fmod(hwidth + vy - deph - hmin, hwidth);
    
        double coef[LAG_PTS];
    
#ifdef LAG_ODD
        int ipos1 = floor((hstar-hmin) * idh);
#else
        int ipos1 = round((hstar-hmin) * idh);
#endif
        const double d_prev1 = LAG_OFFSET + idh * (hstar - (hmin + ipos1 *dh));
        ipos1 -= LAG_OFFSET;
        lag_basis(d_prev1, coef);
    
        double ftmp = 0.;
        for(int k=0; k<=LAG_ORDER; k++) {
          int idx_ipos1 = (s_nvymax + ipos1 + k) % s_nvymax;
          ftmp += coef[k] * fn_(ix, iy, ivx, idx_ipos1);
        }
        fnp1_(ix, iy, ivx, ivy) = ftmp;
      }
    }
  };

  void advect_1D_x(Config *conf, const view_4d fn, view_4d fnp1, double dt);
  void advect_1D_y(Config *conf, const view_4d fn, view_4d fnp1, double dt);
  void advect_1D_vx(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt);
  void advect_1D_vy(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt);
  void print_fxvx(Config *conf, const view_4d fn, int iter);

  void advect_1D_x(Config *conf, const view_4d fn, view_4d fnp1, double dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    #if defined( POLICY_3D )
      // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
      MDPolicyType_3D mdpolicy_3d( {{0,0,0}}, {{s_nxmax,s_nymax,s_nvxmax}}, {{TILE_X0,TILE_Y0,TILE_Z0}} );
      Kokkos::parallel_for("md3d_advection_x", mdpolicy_3d, advect_1D_x_functor(conf, fn, fnp1, dt));
    #else
      int s_nvymax = dom->nxmax_[3];
      int elem = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
      Kokkos::parallel_for(elem, advect_1D_x_functor(conf, fn, fnp1, dt));
    #endif
  };

  void advect_1D_y(Config *conf, const view_4d fn, view_4d fnp1, double dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    #if defined( POLICY_3D )
      // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
      MDPolicyType_3D mdpolicy_3d( {{0,0,0}}, {{s_nxmax,s_nymax,s_nvxmax}}, {{TILE_X1,TILE_Y1,TILE_Z1}} );
      Kokkos::parallel_for("md3d_advection_y", mdpolicy_3d, advect_1D_y_functor(conf, fn, fnp1, dt));
    #else
      int s_nvymax = dom->nxmax_[3];
      int elem = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
      Kokkos::parallel_for(elem, advect_1D_y_functor(conf, fn, fnp1, dt));
    #endif
  };

  void advect_1D_vx(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    #if defined( POLICY_3D )
      // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
      MDPolicyType_3D mdpolicy_3d( {{0,0,0}}, {{s_nxmax,s_nymax,s_nvxmax}}, {{TILE_X2,TILE_Y2,TILE_Z2}} );
      Kokkos::parallel_for("md3d_advection_vx", mdpolicy_3d, advect_1D_vx_functor(conf, fn, fnp1, ef, dt));
    #else
      int s_nvymax = dom->nxmax_[3];
      int elem = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
      Kokkos::parallel_for(elem, advect_1D_vx_functor(conf, fn, fnp1, ef, dt));
    #endif
  };
  
  void advect_1D_vy(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt) {
    const Domain *dom = &(conf->dom_);
    int s_nxmax  = dom->nxmax_[0];
    int s_nymax  = dom->nxmax_[1];
    int s_nvxmax = dom->nxmax_[2];

    #if defined( POLICY_3D )
      // Construct 3D MDRangePolicy: lower, upper bounds, tile dims provided
      MDPolicyType_3D mdpolicy_3d( {{0,0,0}}, {{s_nxmax,s_nymax,s_nvxmax}}, {{TILE_X3,TILE_Y3,TILE_Z3}} );
      Kokkos::parallel_for("md3d_advection_vy", mdpolicy_3d, advect_1D_vy_functor(conf, fn, fnp1, ef, dt));
    #else
      int s_nvymax = dom->nxmax_[3];
      int elem = s_nxmax * s_nymax * s_nvxmax * s_nvymax;
      Kokkos::parallel_for(elem, advect_1D_vy_functor(conf, fn, fnp1, ef, dt));
    #endif
  };

  void print_fxvx(Config *conf, const view_4d fn, int iter) {
    Domain* dom = &(conf->dom_);
    char filename[128];
    printf("print_fxvx %d\n", iter);
    sprintf(filename, "fxvx%04d.out", iter);
    FILE* fileid = fopen(filename, "w");
    
    const int ivy = dom->nxmax_[3] / 2;
    const int iy = 0;
    
    for(int ivx = 0; ivx < dom->nxmax_[2]; ivx++)
    {
      for(int ix = 0; ix < dom->nxmax_[0]; ix++)
        fprintf(fileid, "%4d %4d %20.13le\n", ix, ivx, fn(ix, iy, ivx, ivy));
     
      fprintf(fileid, "\n");
    }
    
    fclose(fileid);
  };
};

#endif
