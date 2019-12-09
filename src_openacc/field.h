#ifndef __FIELD_H__
#define __FIELD_H__

#include "config.h"
#include "efield.h"
#include "diags.h"
#include "types.h"

/*
 * @param[in] fn
 * @param[out] ef.rho_ (Updated by the integral of fn)
 * @param[out] ef.ex_ (zero initialization)
 * @param[out] ef.ey_ (zero initialization)
 * @param[out] ef.phi_ (zero initialization)
 */
void field_rho(Config *conf, view_4d fn, Efield *ef);
void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

#endif
