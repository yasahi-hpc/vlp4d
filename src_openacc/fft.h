#ifndef __FFT_H__
#define __FFT_H__

/*
 * Simple wrapper for FFT class (only 2D FFT implemented now)
 * CUFFT interface for OpenACC
 * https://www.olcf.ornl.gov/tutorials/mixing-openacc-with-gpu-libraries/
 */

#include <cufft.h>
#include <cassert>
#include <openacc.h>
#include "types.h"

namespace Impl {
  struct FFT {
    private:
      cufftHandle backward_plan_, forward_plan_;

    public:
      // Number of real points in the x1 (x) dimension
      int nx1_; 

      // Number of real points in the x2 (y) dimension
      int nx2_;

      // Number of batches
      int nb_batches_;

      // number of complex points+1 in the x1 (x) dimension
      int nx1h_;

      // number of complex points+1 in the x2 (y) dimension
      int nx2h_;

    FFT(int nx1, int nx2)
      : nx1_(nx1), nx2_(nx2), nb_batches_(1) {
      init();
    }

    FFT(int nx1, int nx2, int nb_batches) 
      : nx1_(nx1), nx2_(nx2), nb_batches_(nb_batches) {
      init();
    }

    virtual ~FFT() {
      cufftDestroy(forward_plan_);
      cufftDestroy(backward_plan_);
    }

    void fft2(float64 *dptr_in, complex64 *dptr_out)
    {
      cufftExecD2Z(forward_plan_, 
                   reinterpret_cast<double *>(dptr_in), 
                   reinterpret_cast<cuDoubleComplex *>(dptr_out));
    }

    void ifft2(complex64 *dptr_in, float64 *dptr_out)
    {
      cufftExecZ2D(backward_plan_, 
                   reinterpret_cast<cuDoubleComplex *>(dptr_in), 
                   reinterpret_cast<double *>(dptr_out));
    }

    private:
    void init() {
      nx1h_ = nx1_/2 + 1;
      nx2h_ = nx2_/2 + 1;

      // Create forward/backward plans
      cufftCreate(&forward_plan_);
      cufftCreate(&backward_plan_);

      assert(nb_batches_ >= 1);

      if(nb_batches_ == 1) {
        cufftPlan2d(&forward_plan_, nx2_, nx1_, CUFFT_D2Z);
        cufftPlan2d(&backward_plan_, nx2_, nx1_, CUFFT_Z2D);
      } else {
        // Batched plan
        int rank = 2;
        int n[2];
        int inembed[2], onembed[2];
        int istride, ostride;
        int idist, odist;
        n[0] = nx2_; n[1] = nx1_;
        idist = nx2_*nx1_;
        odist = nx2_*(nx1h_);

        inembed[0] = nx2_; inembed[1] = nx1_;
        onembed[0] = nx2_; onembed[1] = nx1h_;
        istride = 1; ostride = 1;

        // Forward plan
        cufftPlanMany(&forward_plan_,
                      rank,       // rank
                      n,          // dimensions = {pixels.x, pixels.y}
                      inembed,    // Input size with pitch
                      istride,    // Distance between two successive input elements
                      idist,      // Distance between batches for input array
                      onembed,    // Output size with pitch
                      ostride,    // Distance between two successive output elements
                      odist,      // Distance between batches for output array
                      CUFFT_D2Z,  // Cufft Type
                      nb_batches_); // The number of FFTs executed by this plan

        // Backward plan
        cufftPlanMany(&backward_plan_,
                      rank,       // rank
                      n,          // dimensions = {pixels.x, pixels.y}
                      onembed,    // Input size with pitch
                      ostride,    // Distance between two successive input elements
                      odist,      // Distance between batches for input array
                      inembed,    // Output size with pitch
                      istride,    // Distance between two successive output elements
                      idist,      // Distance between batches for output array
                      CUFFT_Z2D,  // Cufft Type
                      nb_batches_); // The number of FFTs executed by this plan
      }

      // Force cuFFT on OpenACC stream https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/openacc/5-openacc-interoperability.pdf?__blob=publicationFile
      cudaStream_t accStream = (cudaStream_t) acc_get_cuda_stream(acc_async_sync);
      cufftSetStream(forward_plan_, accStream);
      cufftSetStream(backward_plan_, accStream);
    }
  };
};
#endif
