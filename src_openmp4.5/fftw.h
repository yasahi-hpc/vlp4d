#ifndef __FFTW_H__
#define __FFTW_H__

#include <omp.h>
#include <cassert>
#include <iostream>
#include "types.h"
#include "index.h"
#include <fftw3.h>

namespace Impl {
  struct FFT {
    int nx1_, nx2_, nb_batches_;
    int nx1h_, nx2h_;
    fftw_plan forward_c2c_plan_, forward_r2c_plan_;
    fftw_plan backward_c2c_plan_, backward_c2r_plan_;

    // Thread private buffer
    complex64 *dptr_buffer_c_;
    complex64 *thread_private_buffers_nx1h_, *thread_private_buffers_nx2_;
    complex64 *thread_private_buffers_nx2_out_;
    float64   *thread_private_buffers_nx1_r2c_;
    float64   *thread_private_buffers_nx1_c2r_;

    complex_view_3d d_buffer_c_;
    complex_view_2d d_thread_private_buffers_nx1h_, d_thread_private_buffers_nx2_;
    complex_view_2d d_thread_private_buffers_nx2_out_;
    view_2d d_buffers_nx1_r2c_, d_buffers_nx1_c2r_;

    FFT(int nx1, int nx2)
      : nx1_(nx1), nx2_(nx2), nb_batches_(1) {
      init();
    }

    FFT(int nx1, int nx2, int batch)
      : nx1_(nx1), nx2_(nx2), nb_batches_(batch) {
      init();
    }

    virtual ~FFT() {
      fftw_destroy_plan(forward_c2c_plan_);
      fftw_destroy_plan(backward_c2c_plan_);
      fftw_destroy_plan(forward_r2c_plan_);
      fftw_destroy_plan(backward_c2r_plan_);

      deallocate(d_buffer_c_);
      deallocate(d_thread_private_buffers_nx1h_);
      deallocate(d_thread_private_buffers_nx2_);
      deallocate(d_thread_private_buffers_nx2_out_);
      deallocate(d_buffers_nx1_r2c_);
      deallocate(d_buffers_nx1_c2r_);
    }

    void fft(complex64 *dptr_in, complex64 *dptr_out) {
      fftw_complex *in  = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft(forward_c2c_plan_, in, out);
    }

    void fftr2c(float64 *dptr_in, complex64 *dptr_out) {
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft_r2c(forward_r2c_plan_, dptr_in, out);
    }

    void ifft(complex64 *dptr_in, complex64 *dptr_out) {
      fftw_complex *in  = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft(backward_c2c_plan_, in, out);
    }

    void ifftc2r(complex64 *dptr_in, float64 *dptr_out) {
      fftw_complex *in = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_execute_dft_c2r(backward_c2r_plan_, in, dptr_out);
    }

    /* In the host code, we assume LayoutRight (C style)
     */
    void fft2(float64 *dptr_in, complex64 *dptr_out) {
      if(nb_batches_ == 1) {
        fft2_serial(dptr_in, dptr_out);
      }
      else {
        fft2_batch(dptr_in, dptr_out);
      }
    }

    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in(nx1h,nx2,batch)
     * @param[out] dptr_out(nx1,nx2,batch)
     */
    void ifft2(complex64 *dptr_in, float64 *dptr_out) {
      if(nb_batches_ == 1) {
        ifft2_serial(dptr_in, dptr_out);
      }
      else {
        ifft2_batch(dptr_in, dptr_out);
      }
    }

    private:
    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in[nx2,nx1)
     * @param[out] dptr_out[nx2,nx1h]
     */
    void fft2_serial(float64 *dptr_in, complex64 *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float64   *thread_private_buffer_nx1 = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        complex64 *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        complex64 *thread_private_buffer_nx2 = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1_, nx2_);
            thread_private_buffer_nx1[ix1] = dptr_in[idx];
          }
          fftr2c(thread_private_buffer_nx1, thread_private_buffer_nx1h);

          // Transpose [nx2,nx1h] -> [nx1h,nx2]
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1h_);
            dptr_buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          int offset = nx2_ * ix1;
          fft(&dptr_buffer_c_[offset], thread_private_buffer_nx2);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_, nx2_);
            dptr_out[idx] = thread_private_buffer_nx2[ix2];
          }
        }
        #pragma omp barrier
      }
    }

    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in[batch,nx2,nx1]
     * @param[out] dptr_out[batch,nx2,nx1h]
     */
    void fft2_batch(float64 *dptr_in, complex64 *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float64   *thread_private_buffer_nx1  = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        complex64 *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        complex64 *thread_private_buffer_nx2  = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1_, nx2_, nb_batches_);
              thread_private_buffer_nx1[ix1] = dptr_in[idx];
            }

            fftr2c(thread_private_buffer_nx1, thread_private_buffer_nx1h);

          // Transpose [batch,nx2,nx1h] -> [batch,nx1h,nx2]
            for(int ix1=0; ix1 < nx1h_; ix1++) {
              int idx = Index::coord_3D2int(ix2, ix1, ib, nx2_, nx1h_, nb_batches_);
              dptr_buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
            }
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int offset = nx2_ * Index::coord_2D2int(ix1, ib, nx1h_, nb_batches_);
            fft(&dptr_buffer_c_[offset], thread_private_buffer_nx2);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              dptr_out[idx] = thread_private_buffer_nx2[ix2];
            }
          }
        }
        #pragma omp barrier
      }
    }

    /* @brief 2D FFT wrapper for serial case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in[nx2,nx1h]
     * @param[out] dptr_out[nx2,nx1]
     */
    void ifft2_serial(complex64 *dptr_in, float64 *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float64   *thread_private_buffer_nx1 = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        complex64 *thread_private_buffer_nx2 = &thread_private_buffers_nx2_[nx2_*tid];
        complex64 *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_,nx2_);
            thread_private_buffer_nx2[ix2] = dptr_in[idx];
          }
          ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_, nx2_);
            dptr_buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          int offset_in  = nx1h_ * ix2;

          ifftc2r(&dptr_buffer_c_[offset_in], thread_private_buffer_nx1);
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1_, nx2_);
            dptr_out[idx] = thread_private_buffer_nx1[ix1];
          }
        }
        #pragma omp barrier
      }
    }

    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in[batch,nx2,nx1h]
     * @param[out] dptr_out[batch,nx2,nx1]
     */
    void ifft2_batch(complex64 *dptr_in, float64 *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float64   *thread_private_buffer_nx1     = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        complex64 *thread_private_buffer_nx2     = &thread_private_buffers_nx2_[nx2_*tid];
        complex64 *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              thread_private_buffer_nx2[ix2] = dptr_in[idx];
            }
            ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              dptr_buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
            }
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int offset  = nx1h_ * Index::coord_2D2int(ix2, ib, nx2_, nb_batches_);

            ifftc2r(&dptr_buffer_c_[offset], thread_private_buffer_nx1);
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1_, nx2_, nb_batches_);
              dptr_out[idx] = thread_private_buffer_nx1[ix1];
            }
          }
        }
        #pragma omp barrier
      }
    }

    void init() {
      nx1h_ = nx1_/2 + 1;
      nx2h_ = nx2_/2 + 1;

      assert(nb_batches_ >= 1);

      // Initialize fftw
      fftw_complex *c_in, *c_out;
      fftw_complex *c_in_c2r, *c_out_r2c;
      float64      *in, *out;

      c_in = fftw_alloc_complex(nx2_);
      c_out = fftw_alloc_complex(nx2_);
      
      in = fftw_alloc_real(nx1_);
      out = fftw_alloc_real(nx1_+2);
      c_in_c2r = fftw_alloc_complex(nx1h_);
      c_out_r2c = fftw_alloc_complex(nx1h_);

      forward_c2c_plan_  = fftw_plan_dft_1d(nx2_, c_in, c_out, FFTW_FORWARD,  FFTW_ESTIMATE);
      backward_c2c_plan_ = fftw_plan_dft_1d(nx2_, c_out, c_in, FFTW_BACKWARD, FFTW_ESTIMATE);

      forward_r2c_plan_  = fftw_plan_dft_r2c_1d(nx1_, in, c_out_r2c, FFTW_ESTIMATE);
      backward_c2r_plan_ = fftw_plan_dft_c2r_1d(nx1_, c_in_c2r, out, FFTW_ESTIMATE);

      fftw_free(in);   fftw_free(out);
      fftw_free(c_in); fftw_free(c_out);
      fftw_free(c_in_c2r); fftw_free(c_out_r2c);

      // Malloc thread private buffers
      size_t nb_threads=0;
      #pragma omp parallel
      nb_threads = static_cast<size_t>( omp_get_num_threads() );

      std::cout << "nb_threads = " << nb_threads << std::endl;
      size_t nx1 = nx1_, nx2 = nx2_, nx1h = nx1h_, nb_batches = nb_batches_;
      allocate(d_buffer_c_, {nx2, nx1h, nb_batches});
      allocate(d_thread_private_buffers_nx1h_, {nx1h,nb_threads});
      allocate(d_thread_private_buffers_nx2_, {nx2,nb_threads});
      allocate(d_thread_private_buffers_nx2_out_, {nx2,nb_threads});

      allocate(d_buffers_nx1_r2c_, {nx1,  nb_threads});
      allocate(d_buffers_nx1_c2r_, {nx1+2,nb_threads});

      dptr_buffer_c_                  = d_buffer_c_.raw();
      thread_private_buffers_nx1h_    = d_thread_private_buffers_nx1h_.raw();
      thread_private_buffers_nx2_     = d_thread_private_buffers_nx2_.raw();
      thread_private_buffers_nx2_out_ = d_thread_private_buffers_nx2_out_.raw();
      thread_private_buffers_nx1_r2c_ = d_buffers_nx1_r2c_.raw();
      thread_private_buffers_nx1_c2r_ = d_buffers_nx1_c2r_.raw();
    }
  };
};

#endif
