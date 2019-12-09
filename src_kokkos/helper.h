#ifndef __HELPER_H__
#define __HELPER_H__
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "types.h"
#include "index.h"

static void L2norm(view_2d f) {
  const size_t nx  = f.extent_int(0);
  const size_t ny  = f.extent_int(1);
  double norm = 0.;

  Kokkos::parallel_reduce(nx * ny, KOKKOS_LAMBDA (const int i, double& lsum) {
    int2 idx_2D = Index::int2coord_2D(i, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;
    lsum += f(ix, iy) * f(ix, iy);
  }, norm);

  std::stringstream ss;
  ss << "L2 Norm of " << f.label() << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

static void printAll(view_2d f) {
  const size_t nx  = f.extent_int(0);
  const size_t ny  = f.extent_int(1);

  view_2d::HostMirror h_f = Kokkos::create_mirror_view(f);
  Kokkos::deep_copy(h_f, f);

  std::stringstream ss;
  ss << f.label() << ".dat";
  std::ofstream outfile;
  outfile.open(ss.str(), std::ios::out);

  for(int iy=0; iy<ny; iy++) {
    for(int ix=0; ix<nx; ix++) {
      outfile << h_f(ix,iy) << " ";
    }
    outfile << "\n";
  }
}

static void L2norm(view_4d f) {
  const size_t nx  = f.extent_int(0);
  const size_t ny  = f.extent_int(1);
  const size_t nvx = f.extent_int(2);
  const size_t nvy = f.extent_int(3);
  double norm = 0.;

  Kokkos::parallel_reduce(nx * ny * nvx * nvy, KOKKOS_LAMBDA (const int i, double& lsum) {
    int4 idx_4D = Index::int2coord_4D(i, nx, ny, nvx, nvy);
    int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;
    lsum += f(ix, iy, ivx, ivy) * f(ix, iy, ivx, ivy);
  }, norm);

  std::stringstream ss;
  ss << "L2 Norm of " << f.label() << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

static void printAll(view_4d f) {
  const size_t nx  = f.extent_int(0);
  const size_t ny  = f.extent_int(1);
  const size_t nvx = f.extent_int(2);
  const size_t nvy = f.extent_int(3);

  view_4d::HostMirror h_f = Kokkos::create_mirror_view(f);
  Kokkos::deep_copy(h_f, f);

  std::stringstream ss;
  ss << f.label() << ".dat";
  std::ofstream outfile;
  outfile.open(ss.str(), std::ios::out);

  for(int ivy=0; ivy<nvy; ivy++) {
    for(int ivx=0; ivx<nvx; ivx++) {
      for(int iy=0; iy<ny; iy++) {
        for(int ix=0; ix<nx; ix++) {
          outfile << h_f(ix,iy,ivx,ivy) << " ";
        }
      }
    }
  }
}

#endif
