#include <cmath>
#include <cstring>

#include <cudnn.h>

#include "common.h"

/**
 * Performs a convolution (really cross-correlation) on the CPU.
 */
void ConvolutionHost(float *x, int batch_size, int c_in, int h_in, int w_in,
    float *w, int c_out, int kernel_h, int kernel_w, int stride_h, int stride_w,
    float *y) {
  int h_out = (h_in - kernel_h) / stride_h + 1;
  int w_out = (w_in - kernel_w) / stride_w + 1;

  // Set the output to zero so we can accumulate
  memset(y, 0, batch_size*c_out*h_out*w_out*sizeof(float));
  
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < c_out; ++j) {
      for (int k = 0; k < h_out; ++k) {
        for (int l = 0; l < w_out; ++l) {

          // Compute the convolution for this output pixel
          for (int m = 0; m < c_in; ++m) {
            for (int n = 0; n < kernel_h; ++n) {
              for (int o = 0; o < kernel_w; ++o) {
                y[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] +=
                  w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w + n*kernel_w + o] *
                  x[i*c_in*h_in*w_in + m*h_in*w_in + ((k*stride_h)+n)*w_in + ((l*stride_w)+o)];
              }
            }
          } 
        }
      }
    }
  }
}

__global__ void ConvolutionKernel(float *x, int batch_size, int c_in, int h_in, int w_in,
    float *w, int c_out, int kernel_h, int kernel_w, int stride_h, int stride_w,
    float *y, int h_out, int w_out) {
  int i = blockIdx.x; // datum index
  int j = blockIdx.y; // feature map index

  for (int k = threadIdx.y; k < h_out; k += blockDim.y) {
    for (int l = threadIdx.x; l < w_out; l += blockDim.x) {

      // Compute the convolution for this output pixel
      for (int m = 0; m < c_in; ++m) {
        for (int n = 0; n < kernel_h; ++n) {
          for (int o = 0; o < kernel_w; ++o) {
            y[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] +=
              w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w + n*kernel_w + o] *
              x[i*c_in*h_in*w_in + m*h_in*w_in + ((k*stride_h)+n)*w_in + ((l*stride_w)+o)];
          }
        }
      }
    }
  }
}

void ConvolutionDevice(float *x, int batch_size, int c_in, int h_in, int w_in,
    float *w, int c_out, int kernel_h, int kernel_w, int stride_h, int stride_w,
    float *y, cudaStream_t stream) {
  int h_out = (h_in - kernel_h) / stride_h + 1;
  int w_out = (w_in - kernel_w) / stride_w + 1;
  CUDA_CALL(cudaMemsetAsync(y, 0, batch_size*c_out*h_out*w_out*sizeof(float), stream));
  ConvolutionKernel<<<dim3(batch_size, c_out), dim3(32, 32), 0, stream>>>(x,
      batch_size, c_in, h_in, w_in, w, c_out, kernel_h, kernel_w, stride_h,
      stride_w, y, h_out, w_out);
}

int main() {
  cudnnHandle_t handle;
  CUDNN_CALL(cudnnCreate(&handle));

  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreate(&stream));

  CUDNN_CALL(cudnnSetStream(handle, stream));
  
  //
  /// Set dimensions for the convolution
  //

  // Kernel dims - we don't support padding in this example
  int kernel_h = 3;
  int kernel_w = 3;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 1;
  int stride_w = 1;

  // Input dims
  int n = 32;
  int h_in = 227;
  int w_in = 227;
  int c_in = 3;

  // Output dims
  int h_out = (h_in + 2*pad_h - kernel_h) / stride_h + 1;
  int w_out = (w_in + 2*pad_w - kernel_w) / stride_w + 1;
  int c_out = 32;

  //
  /// Setup data & filters for the convolution
  //
  
  int filter_size = c_out*c_in*kernel_h*kernel_w;
  int images_size = n*c_in*h_in*w_in;
  int output_size = n*c_out*h_out*w_out;
  float *filters = new float[filter_size];
  float *images = new float[images_size];
  float *output = new float[output_size];
  SetIncremental(filters, filter_size);
  SetIncremental(images, images_size);

#ifdef DEBUG
  cout << "Images: ";
  PrintTensor(images, n, c_in, h_in, w_in);
  cout << "Filters: ";
  PrintTensor(filters, c_out, c_in, kernel_h, kernel_w);
#endif
  
  // Setup device version of input, output, and filters
  float *filters_on_device, *images_on_device, *output_on_device;
  CUDA_CALL(cudaMalloc(&filters_on_device, filter_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&images_on_device, images_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&output_on_device, output_size*sizeof(float)));
  CUDA_CALL(cudaMemcpy(
          filters_on_device, filters,
          filter_size*sizeof(float),
          cudaMemcpyHostToDevice
          ));
  CUDA_CALL(cudaMemcpy(
          images_on_device,
          images, images_size*sizeof(float),
          cudaMemcpyHostToDevice
          ));

  //
  /// Setup parameters for cudnn call
  //

  // Setup alpha/beta
  float alpha = 1.f, beta = 0.f;
  
  // Setup input/output tensor descriptors
  cudnnTensorDescriptor_t x_desc, y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_in, h_in, w_in));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_out, h_out, w_out));
  
  // Setup filter descriptor
  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
          CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          c_out, c_in, kernel_h, kernel_w));
    
  // Setup convolution meta-data
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc,
          pad_h, pad_w, stride_h, stride_w, 1, 1,
          CUDNN_CROSS_CORRELATION,
          CUDNN_DATA_FLOAT));

  // Setup & allocate workspace
  size_t workspace_size = 0;
  void *workspace_on_device;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc,
          w_desc, conv_desc, y_desc, conv_algo, &workspace_size));
  CUDA_CALL(cudaMalloc(&workspace_on_device, workspace_size));

  // Run the convolution
  auto t1 = GetTime();
  CUDNN_CALL(cudnnConvolutionForward(
          handle,
          &alpha,
          x_desc,
          images_on_device,
          w_desc,
          filters_on_device,
          conv_desc,
          conv_algo,
          workspace_on_device,
          workspace_size, 
          &beta, 
          y_desc,
          output_on_device
          ));

  CUDA_CALL(cudaStreamSynchronize(stream));
  float total_seconds = ElapsedTime(t1, GetTime());
  cout << "CUDNN FPS: " << n / total_seconds << endl;
  
#ifdef DEBUG
  cout << "Device Output: ";
  PrintTensor(output_on_device, n, c_out, h_out, w_out);
#endif
  
  // Do the host-side convolution
  t1 = GetTime();
  ConvolutionHost(
      images,
      n,
      c_in,
      h_in,
      w_in,
      filters,
      c_out,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      output);

  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Host FPS: " << n / total_seconds << endl;
  
#ifdef DEBUG
  cout << "Output: ";
  PrintTensor(output, n, c_out, h_out, w_out);
#endif
  
  // Verify the results
  VerifyResults(output, output_on_device, n*c_out*h_out*w_out);

  // Run the device convolution
  t1 = GetTime();
  ConvolutionDevice(
      images_on_device,
      n,
      c_in,
      h_in,
      w_in,
      filters_on_device,
      c_out,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      output_on_device,
      stream
      );

  CUDA_CALL(cudaStreamSynchronize(stream));
  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Device FPS: " << n / total_seconds << endl;

  // Verify the results
  VerifyResults(output, output_on_device, n*c_out*h_out*w_out);
  
  // clean up
  CUDA_CALL(cudaFree(workspace_on_device));
  CUDA_CALL(cudaFree(filters_on_device));
  CUDA_CALL(cudaFree(images_on_device));
  CUDA_CALL(cudaFree(output_on_device));
  delete[] filters;
  delete[] images;
  delete[] output;
  CUDNN_CALL(cudnnDestroy(handle));
}
