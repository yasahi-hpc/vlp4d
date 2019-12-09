#ifndef __TIMESTEP_H__
#define __TIMESTEP_H__

#include "efield.h"
#include "diags.h"
#include "types.h"

void onetimestep(Config *conf, view_4d fn, view_4d fnp1, Efield *ef, Diags *dg, int iter);

#endif
