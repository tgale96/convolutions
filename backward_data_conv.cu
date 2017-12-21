#include <cmath>
#include <cstring>

#include <algorithm>

#include <cudnn.h>

#include "common.h"

/**
 * Compute the gradient of the convolution w.r.t. the input data.
 */
void ConvolutionBackwardDataHost(float *dy, int batch_size, int c_out, int h_out, int w_out,
    float *w, int c_in, int kernel_h, int kernel_w, int stride_h, int stride_w, float *dx,
    int h_in, int w_in) {

  // Set the output to zero so we can accumulate
  memset(dx, 0, batch_size*c_in*h_in*w_in*sizeof(float));
  
  for (int i = 0; i < batch_size; ++i) {
    for (int m = 0; m < c_in; ++m) {
      for (int p = 0; p < h_in; ++p) {
        for (int q = 0; q < w_in; ++q) {
          // cout << "calculating index: " << p << ", " << q << endl;
          int k_start = max(0, (int)floor((p - kernel_h) / float(stride_h)) + 1);
          int k_end = min(h_out, p / stride_h + 1);
          int l_start = max(0, (int)floor((q - kernel_w) / float(stride_w)) + 1);
          int l_end = min(w_out, q / stride_w + 1);

          // cout << "iter on k: " << k_start << " -> " << k_end << endl;
          // cout << "iter on l: " << l_start << " -> " << l_end << endl;
          
          // compute the gradient of this input pixel
          for (int j = 0; j < c_out; ++j) {
            for (int k = k_start; k < k_end; ++k) {
              for (int l = l_start; l < l_end; ++l) {
                // cout << "accumulating: " <<
                //   dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l];
                // cout << " * " <<
                //   w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w +
                //       (p - (k*stride_h))*kernel_w + (q - (l * stride_w))]
                //      << " = " << dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] *
                //   w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w +
                //       (p - (k*stride_h))*kernel_w + (q - (l * stride_w))] << endl;
                // cout << "filter index: " << p - (k*stride_h) << ", " << q - (l*stride_w) << endl;
                
                dx[i*c_in*h_in*w_in + m*h_in*w_in + p*w_in + q] +=
                  dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] *
                  w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w +
                      (p - (k*stride_h))*kernel_w + (q - (l * stride_w))];
              }
            }
          }
        }
      }
    }
  }
}

__global__ void ConvolutionBackwardDataKernel(float *dy, int batch_size, int c_out,
    int h_out, int w_out, float *w, int c_in, int kernel_h, int kernel_w, int stride_h,
    int stride_w, float *dx, int h_in, int w_in) {
  int i = blockIdx.x; // sample index
  int m = blockIdx.y; // channel_in index

  for (int p = threadIdx.y; p < h_in; p += blockDim.y) {
    for (int q = threadIdx.x; q < w_in; q += blockDim.x) {
      int k_start = max(0, (int)floor((p - kernel_h) / float(stride_h)) + 1);
      int k_end = min(h_out, p / stride_h + 1);
      int l_start = max(0, (int)floor((q - kernel_w) / float(stride_w)) + 1);
      int l_end = min(w_out, q / stride_w + 1);

      // compute the gradient of this input pixel
      for (int j = 0; j < c_out; ++j) {
        for (int k = k_start; k < k_end; ++k) {
          for (int l = l_start; l < l_end; ++l) {
            dx[i*c_in*h_in*w_in + m*h_in*w_in + p*w_in + q] +=
              dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] *
              w[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w +
                  (p - (k*stride_h))*kernel_w + (q - (l * stride_w))];
          }
        }
      }
    }
  }  
}

void ConvolutionBackwardData(float *dy, int batch_size, int c_out, int h_out, int w_out,
    float *w, int c_in, int kernel_h, int kernel_w, int stride_h, int stride_w, float *dx,
    int h_in, int w_in, cudaStream_t stream) {
  CUDA_CALL(cudaMemsetAsync(dx, 0, batch_size*c_in*h_in*w_in*sizeof(float), stream));
  ConvolutionBackwardDataKernel<<<dim3(batch_size, c_in), dim3(32, 32), 0, stream>>>(
      dy, batch_size, c_out, h_out, w_out, w, c_in, kernel_h, kernel_w, stride_h, stride_w,
      dx, h_in, w_in);
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
  int image_grad_size = n*c_in*h_in*w_in;
  int output_grad_size = n*c_out*h_out*w_out;
  float *filters = new float[filter_size];
  float *image_grad = new float[image_grad_size];
  float *output_grad = new float[output_grad_size];
  SetIncremental(filters, filter_size);
  SetIncremental(output_grad, output_grad_size);

#ifdef DEBUG
  cout << "Filters: ";
  PrintTensor(filters, c_out, c_in, kernel_h, kernel_w);
  cout << "Output Grad: ";
  PrintTensor(output_grad, n, c_out, h_out, w_out);
#endif
  
  // Setup device version of input, output, and filters
  float *filters_on_device, *image_grad_on_device, *output_grad_on_device;
  CUDA_CALL(cudaMalloc(&filters_on_device, filter_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&image_grad_on_device, image_grad_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&output_grad_on_device, output_grad_size*sizeof(float)));
  CUDA_CALL(cudaMemcpy(
          filters_on_device,
          filters, filter_size*sizeof(float),
          cudaMemcpyHostToDevice
          ));
  CUDA_CALL(cudaMemcpy(
          output_grad_on_device,
          output_grad, output_grad_size*sizeof(float),
          cudaMemcpyHostToDevice
          ));

  //
  /// Setup parameters for cudnn call
  //

  // Setup alpha/beta
  float alpha = 1.f, beta = 0.f;

  // Setup input and output grad tensor descriptors
  cudnnTensorDescriptor_t dy_desc, dx_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_out, h_out, w_out));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(dx_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_in, h_in, w_in));

  // Setup filter descriptor
  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(w_desc,
          CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          c_out, c_in, kernel_h, kernel_w));

  // Setup convolution meta-data
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionBwdDataAlgo_t conv_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc,
          pad_h, pad_w, stride_h, stride_w, 1, 1,
          CUDNN_CROSS_CORRELATION,
          CUDNN_DATA_FLOAT));

  // Setup & allocate workspace
  size_t workspace_size = 0;
  void *workspace_on_device;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, 
          dy_desc, conv_desc, dx_desc, conv_algo, &workspace_size));
  CUDA_CALL(cudaMalloc(&workspace_on_device, workspace_size));
  
  // Run the backward convolution w.r.t. the data
  auto t1 = GetTime();
  CUDNN_CALL(cudnnConvolutionBackwardData(
          handle,
          &alpha,
          w_desc,
          filters_on_device,
          dy_desc,
          output_grad_on_device,
          conv_desc,
          conv_algo,
          workspace_on_device,
          workspace_size,
          &beta,
          dx_desc,
          image_grad_on_device
          ));

  CUDA_CALL(cudaStreamSynchronize(stream));
  float total_seconds = ElapsedTime(t1, GetTime());
  cout << "CUDNN FPS: " << n / total_seconds << endl;

#ifdef DEBUG
  cout << "Device Data Grad:";
  PrintTensor(image_grad_on_device, n, c_in, h_in, w_in);
#endif

  // Do the host-side backward convolution w.r.t. the data
  t1 = GetTime();
  ConvolutionBackwardDataHost(
      output_grad,
      n,
      c_out,
      h_out,
      w_out,
      filters,
      c_in,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      image_grad,
      h_in,
      w_in
      );

  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Host FPS: " << n / total_seconds << endl;

#ifdef DEBUG
  cout << "Data Grad: ";
  PrintTensor(image_grad, n, c_in, h_in, w_in);
#endif

  // Verify the results
  VerifyResults(image_grad, image_grad_on_device, n*c_in*h_in*w_in);

  // Do the device-side backward convolution w.r.t. the data
  t1 = GetTime();
  ConvolutionBackwardData(
      output_grad_on_device,
      n,
      c_out,
      h_out,
      w_out,
      filters_on_device,
      c_in,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      image_grad_on_device,
      h_in,
      w_in,
      stream
      );

  CUDA_CALL(cudaStreamSynchronize(stream));
  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Device FPS: " << n / total_seconds << endl;

  // Verify the results
  VerifyResults(image_grad, image_grad_on_device, n*c_in*h_in*w_in);
  
  // clean up
  delete[] filters;
  delete[] image_grad;
  delete[] output_grad;
  CUDA_CALL(cudaFree(workspace_on_device));
  CUDA_CALL(cudaFree(filters_on_device));
  CUDA_CALL(cudaFree(image_grad_on_device));
  CUDA_CALL(cudaFree(output_grad_on_device));
  CUDNN_CALL(cudnnDestroy(handle));
}
