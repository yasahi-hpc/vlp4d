#ifndef __INIT_H__
#define __INIT_H__

#include "efield.h"
#include "diags.h"
#include "view_nd.h"

void init(const char *file, Config *conf, view_4d &fn, view_4d &fnp1, Efield **ef, Diags **dg);
void finalize(Config *conf, view_4d &fn, view_4d &fnp1, Efield **ef, Diags **dg);

#endif
