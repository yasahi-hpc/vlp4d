#ifndef __INDEX_H__
#define __INDEX_H__

#include <Kokkos_Core.hpp>

namespace Index {
#if ! defined( KOKKOS_ENABLE_CUDA )
  // For Layout right (C layout)
  KOKKOS_INLINE_FUNCTION 
  int2 int2coord_2D(int i, int n1, int n2) {
    int j1 = i/n2, j2 = i%n2;
    return make_int2(j1, j2);
  }

  KOKKOS_INLINE_FUNCTION 
  int3 int2coord_3D(int i, int n1, int n2, int n3) {
    int j12 = i/n3, j3 = i%n3;
    int j1 = j12/n2, j2 = j12%n2;
    return make_int3(j1, j2, j3);
  }
  
  KOKKOS_INLINE_FUNCTION 
  int4 int2coord_4D(int i, int n1, int n2, int n3, int n4) {
    int j123 = i/n4, j4 = i%n4;
    int j12 = j123/n3, j3 = j123%n3;
    int j1 = j12/n2, j2 = j12%n2;
    return make_int4(j1, j2, j3, j4);
  }
#else
  // For Layout left (Fortan layout)
  KOKKOS_INLINE_FUNCTION 
  int2 int2coord_2D(int i, int n1, int n2) {
    int j1 = i%n1, j2 = i/n1;
    return make_int2(j1, j2);
  }

  KOKKOS_INLINE_FUNCTION 
  int3 int2coord_3D(int i, int n1, int n2, int n3) {
    int j12 = i%(n1*n2), j3 = i/(n1*n2);
    int j1 = j12%n1, j2 = j12/n1;
    return make_int3(j1, j2, j3);
  }
  
  KOKKOS_INLINE_FUNCTION 
  int4 int2coord_4D(int i, int n1, int n2, int n3, int n4) {
    int j123 = i%(n1*n2*n3), j4 = i/(n1*n2*n3);
    int j12 = j123%(n1*n2), j3 = j123/(n1*n2);
    int j1 = j12%n1, j2 = j12/n1;
    return make_int4(j1, j2, j3, j4);
  }
#endif

  KOKKOS_INLINE_FUNCTION 
  int coord_2D2int(int i1, int i2, int n1, int n2) {
    int idx = i1 + n1*i2;
    return idx;
  }

  KOKKOS_INLINE_FUNCTION 
  int coord_3D2int(int i1, int i2, int i3, int n1, int n2, int n3) {
    int idx = i1 + i2*n1 + i3*n1*n2;
    return idx;
  }
  
  KOKKOS_INLINE_FUNCTION 
  int coord_4D2int(int i1, int i2, int i3, int i4, int n1, int n2, int n3, int n4) {
    int idx = i1 + i2*n1 + i3*n1*n2 + i4*n1*n2*n3;
    return idx;
  }
};

#endif
