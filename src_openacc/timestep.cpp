#include "timestep.h"
#include "advect.h"
#include "field.h"

void onetimestep(Config *conf, view_4d fn, view_4d fnp1, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);

  advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);
  field_rho(conf, fn, ef);
  field_poisson(conf, ef, dg, iter);
  advect_1D_vx(conf, fn, fnp1, ef, dom->dt_);
  advect_1D_vy(conf, fnp1, fn, ef, dom->dt_);
  advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);

  if(dom->fxvx_)
    print_fxvx(conf, fn, iter);
}
