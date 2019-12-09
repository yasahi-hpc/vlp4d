#include "config.h"
#include "efield.h"
#include "view_nd.h"

void advect_1D_x(Config *conf, const view_4d fn, view_4d fnp1, double dt);
void advect_1D_y(Config *conf, const view_4d fn, view_4d fnp1, double dt);
void advect_1D_vx(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt);
void advect_1D_vy(Config *conf, const view_4d fn, view_4d fnp1, Efield *ef, double dt);
void print_fxvx(Config *conf, const view_4d fn, int iter);
void debug_1D_y(Config *conf, const view_4d fn);
