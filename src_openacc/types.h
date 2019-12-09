#ifndef __TYPES_H__
#define __TYPES_H__

#include <complex>
#include "view_nd.h"

// Call double as float64 for better consistency
typedef double float64;
typedef std::complex<double> complex64;

typedef view_nd<double, 1> view_1d;
typedef view_nd<double, 2> view_2d;
typedef view_nd<double, 4> view_4d;

typedef view_nd<complex64, 1> complex_view_1d;
typedef view_nd<complex64, 2> complex_view_2d;
typedef view_nd<complex64, 3> complex_view_3d;

typedef const view_nd<const double, 1> cview_1d;
typedef const view_nd<const double, 2> cview_2d;
typedef const view_nd<const double, 4> cview_4d;

#if ! defined( ENABLE_OPENACC )
struct int2 {
  int x, y;
};

static inline int2 make_int2(int x, int y) {
  int2 t; t.x = x; t.y= y; return t;
};

struct int3 {
  int x, y, z;
};

static inline int3 make_int3(int x, int y, int z) {
  int3 t; t.x = x; t.y = y; t.z = z; return t;
};

struct int4 {
  int x, y, z, w;
};

static inline int4 make_int4(int x, int y, int z, int w) {
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
};
#endif

#endif
