#ifndef __CUDA_HELPER_H__
#define __CUDA_HELPER_H__

#include <cstdlib>
#include <string>
             
#define SafeCudaCall(call) CheckCudaCall(call, #call, __FILE__, __LINE__)
template <typename T>
void CheckCudaCall(T command, const char * commandName, const char * fileName, int line) {
  if(command) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
            fileName, line, (unsigned int)command, commandName);
           
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#endif
