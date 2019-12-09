#include "timestep.h"
#include "advection_funtors.h"
#include "field.h"
#include "helper.h"

void onetimestep(Config *conf, view_4d fn, view_4d fnp1, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);

  Advection::advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  Advection::advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);
  field_rho(conf, fn, ef);
  field_poisson(conf, ef, dg, iter);
  Advection::advect_1D_vx(conf, fn, fnp1, ef, dom->dt_);
  Advection::advect_1D_vy(conf, fnp1, fn, ef, dom->dt_);
  Advection::advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  Advection::advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);

  if(dom->fxvx_)
    Advection::print_fxvx(conf, fn, iter);
}
