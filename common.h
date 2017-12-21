#include <chrono>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace chrono;

#define CUDA_CALL(code)                                  \
  do {                                                   \
    cudaError_t status = code;                           \
    if (status != cudaSuccess) {                         \
      string file = __FILE__;                            \
      string line = to_string(__LINE__);                 \
      string error = "[" + file + ":" + line +           \
        "]: CUDA error \"" +                             \
        cudaGetErrorString(status) + "\"";               \
      throw runtime_error(error);                        \
    }                                                    \
  } while (0)

#define CUDNN_CALL(code)                                  \
  do {                                                    \
    cudnnStatus_t status = code;                          \
    if (status != CUDNN_STATUS_SUCCESS) {                 \
      string file = __FILE__;                             \
      string line = to_string(__LINE__);                  \
      string error = "[" + file + ":" + line +            \
        "]: CUDNN error \"" +                             \
        cudnnGetErrorString(status) + "\"";               \
      throw runtime_error(error);                         \
    }                                                     \
  } while (0)

/** 
 * Set each element to its index (% 10 to keep the results reasonable) * .01
 */
void SetIncremental(float *data, int n) {
  for (int i = 0; i < n; i++) data[i] = (i % 10 + 1)*.01f;
}

/**
 * Prints tensor
 */
void PrintTensor(float *ptr, int n, int c, int h, int w) {
  cout << n << "x" << c << "x" << h << "x" << w << endl;
  vector<float> tmp(n*c*h*w, 0);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(
          tmp.data(), ptr,
          tmp.size()*sizeof(float),
          cudaMemcpyDefault
          ));
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      for (int k = 0; k < h; ++k) {
        cout << "[ ";
        for (int l = 0; l < w; ++l) {
          cout << tmp[i*c*h*w + j*h*w + k*w + l] << "\t";
        }
        cout << "]," << endl;
      }
    }
  }
}

/**
 * Verifies the two arrays are identical
 */
void VerifyResults(float *a, float *b, int n) {
  vector<float> a_tmp(n, 0), b_tmp(n, 0);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemcpy(
          a_tmp.data(), a,
          a_tmp.size()*sizeof(float),
          cudaMemcpyDefault));
  CUDA_CALL(cudaMemcpy(
          b_tmp.data(), b,
          b_tmp.size()*sizeof(float),
          cudaMemcpyDefault));

  for (int i = 0; i < n; ++i) {
    float abs_diff = abs(a_tmp[i] - b_tmp[i]);
    float tol = .005 * abs(a_tmp[i]);
    if (abs_diff > tol) {
      cout << "non-zero diff at " + to_string(i) + ": " +
        to_string(abs_diff) << endl;
      cout << a_tmp[i] << " v. " << b_tmp[i] << endl;
#ifndef DEBUG
      // If not debugging, die on error
      throw exception();
#endif
    }
  }
}

/**
 * Get a high resolution time point
 */
high_resolution_clock::time_point GetTime() {
  return high_resolution_clock::now();
}

/**
 * Get elapsed time in seconds
 */
float ElapsedTime(high_resolution_clock::time_point start,
    high_resolution_clock::time_point end) {
  return float(duration_cast<nanoseconds>(end - start).count()) / 1000000000;
}
