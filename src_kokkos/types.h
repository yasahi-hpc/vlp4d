#ifndef __TYPES_H__
#define __TYPES_H__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <array>
typedef Kokkos::DefaultExecutionSpace execution_space;

// New directive for vectorization with OpenMP background
#if ! defined( KOKKOS_ENABLE_CUDA )
  #define KOKKOS_SIMD _Pragma("omp simd")
#else
  #define KOKKOS_SIMD
#endif

// Call double as float64 for better consistency
typedef double float64;
typedef Kokkos::complex<double> complex64;
// const complex64 I = complex64(0., 1.); // for some reason, it does not work for GPU version

// Multidimensional view types
typedef Kokkos::View<float64*, execution_space> view_1d;
typedef Kokkos::View<float64**, execution_space> view_2d;
typedef Kokkos::View<float64****, execution_space> view_4d;
typedef view_1d::HostMirror view_1d_host;

typedef Kokkos::View<complex64*, execution_space> complex_view_1d;
typedef Kokkos::View<complex64**, execution_space> complex_view_2d;
typedef Kokkos::View<complex64***, execution_space> complex_view_3d;

// Used for 3D loops
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_3D;

template <unsigned ND> using coord_t = std::array<int, ND>;
template <unsigned ND> using shape_t = std::array<int, ND>;

struct double_pair {
  double x, y;
  KOKKOS_INLINE_FUNCTION
  double_pair(double xinit, double yinit) 
    : x(xinit), y(yinit) {};

  KOKKOS_INLINE_FUNCTION
  double_pair()
    : x(0.), y(0.) {};

  KOKKOS_INLINE_FUNCTION
  double_pair& operator += (const double_pair& src) {
    x += src.x;
    y += src.y;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile double_pair& operator += (const volatile double_pair& src) volatile {
    x += src.x;
    y += src.y;
    return *this;
  }
};

#if ! defined( KOKKOS_ENABLE_CUDA )
struct int2 {
  int x, y;
};

KOKKOS_INLINE_FUNCTION
int2 make_int2(int x, int y) {
  int2 t; t.x = x; t.y= y; return t;
};

struct int3 {
  int x, y, z;
};

KOKKOS_INLINE_FUNCTION
int3 make_int3(int x, int y, int z) {
  int3 t; t.x = x; t.y = y; t.z = z; return t;
};

struct int4 {
  int x, y, z, w;
};

KOKKOS_INLINE_FUNCTION
int4 make_int4(int x, int y, int z, int w) {
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
};

#endif

#endif
