#include <cudnn.h>

#include "common.h"

/**
 * Calculates the gradient w.r.t the weights for a convolution on the CPU.
 */
void ConvolutionBackwardFilterHost(float *x, int batch_size, int c_in, int h_in, int w_in,
    float *dy, int c_out, int h_out, int w_out, float *dw, int kernel_h, int kernel_w,
    int stride_h, int stride_w) {

  // Set the output to zero so we can accumulate
  memset(dw, 0, c_out*c_in*kernel_h*kernel_w*sizeof(float));

  for (int j = 0; j < c_out; ++j) {
    for (int m = 0; m < c_in; ++m) {
      for (int n = 0; n < kernel_h; ++n) {
        for (int o = 0; o < kernel_w; ++o) {

          // Compute the gradient of this filter weight
          for (int i = 0; i < batch_size; ++i) {
            for (int k = 0; k < h_out; ++k) {
              for (int l = 0; l < w_out; ++l) {
                dw[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w + n*kernel_w + o] +=
                  dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] *
                  x[i*c_in*h_in*w_in + m*h_in*w_in + ((k*stride_h)+n)*w_in + ((l*stride_w)+o)];
              }
            }
          }
        }
      }
    }
  }
}

__global__ void ConvolutionBackwardFilterKernel(float *x, int batch_size, int c_in,
    int h_in, int w_in, float *dy, int c_out, int h_out, int w_out, float *dw,
    int kernel_h, int kernel_w, int stride_h, int stride_w) {
  int j = blockIdx.x; // filter index

  for (int m = threadIdx.z; m < c_in; m += blockDim.z) {
    for (int n = threadIdx.y; n < kernel_h; n += blockDim.y) {
      for (int o = threadIdx.x; o < kernel_w; o += blockDim.x) {

        // Compute the gradient of this filter weight
        for (int i = 0; i < batch_size; ++i) {
          for (int k = 0; k < h_out; ++k) {
            for (int l = 0; l < w_out; ++l) {
              dw[j*c_in*kernel_h*kernel_w + m*kernel_h*kernel_w + n*kernel_w + o] +=
                dy[i*c_out*h_out*w_out + j*h_out*w_out + k*w_out + l] *
                x[i*c_in*h_in*w_in + m*h_in*w_in + ((k*stride_h)+n)*w_in + ((l*stride_w)+o)];
            }
          }
        }
      }
    }
  }
}

void ConvolutionBackwardFilter(float *x, int batch_size, int c_in, int h_in, int w_in,
    float *dy, int c_out, int h_out, int w_out,
    float *dw, int kernel_h, int kernel_w, int stride_h, int stride_w, cudaStream_t stream) {
  CUDA_CALL(cudaMemsetAsync(dw, 0, c_out*c_in*kernel_h*kernel_w*sizeof(float), stream));
  ConvolutionBackwardFilterKernel<<<c_out, dim3(8, 8, 8), 0, stream>>>(x, batch_size, c_in,
      h_in, w_in, dy, c_out, h_out, w_out, dw, kernel_h, kernel_w, stride_h, stride_w);
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
  
  int filter_grad_size = c_out*c_in*kernel_h*kernel_w;
  int images_size = n*c_in*h_in*w_in;
  int output_grad_size = n*c_out*h_out*w_out;
  float *filter_grad = new float[filter_grad_size];
  float *images = new float[images_size];
  float *output_grad = new float[output_grad_size];
  SetIncremental(images, images_size);
  SetIncremental(output_grad, output_grad_size);

#ifdef DEBUG
  cout << "Images: ";
  PrintTensor(images, n, c_in, h_in, w_in);
  cout << "Output Grad: ";
  PrintTensor(output_grad, n, c_out, h_out, w_out);
#endif
  
  // Setup device version of input, output, and filters
  float *filter_grad_on_device, *images_on_device, *output_grad_on_device;
  CUDA_CALL(cudaMalloc(&filter_grad_on_device, filter_grad_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&images_on_device, images_size*sizeof(float)));
  CUDA_CALL(cudaMalloc(&output_grad_on_device, output_grad_size*sizeof(float)));
  CUDA_CALL(cudaMemcpy(
          images_on_device,
          images, images_size*sizeof(float),
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
  cudnnTensorDescriptor_t x_desc, dy_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_in, h_in, w_in));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc,
          CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          n, c_out, h_out, w_out));

  // Setup filter grad descriptor
  cudnnFilterDescriptor_t dw_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&dw_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(dw_desc,
          CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          c_out, c_in, kernel_h, kernel_w));

  // Setup backward conv meta-data
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionBwdFilterAlgo_t conv_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc,
          pad_h, pad_w, stride_h, stride_w, 1, 1,
          CUDNN_CROSS_CORRELATION,
          CUDNN_DATA_FLOAT));

  // Setup & allocate workspace
  size_t workspace_size = 0;
  void *workspace_on_device;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc,
          dy_desc, conv_desc, dw_desc, conv_algo, &workspace_size));
  CUDA_CALL(cudaMalloc(&workspace_on_device, workspace_size));
  
  // Run the backward covolution w.r.t the weights
  auto t1 = GetTime();
  CUDNN_CALL(cudnnConvolutionBackwardFilter(
          handle,
          &alpha,
          x_desc,
          images_on_device,
          dy_desc,
          output_grad_on_device,
          conv_desc,
          conv_algo,
          workspace_on_device,
          workspace_size,
          &beta,
          dw_desc,
          filter_grad_on_device
          ));

  CUDA_CALL(cudaStreamSynchronize(stream));
  float total_seconds = ElapsedTime(t1, GetTime());
  cout << "CUDNN FPS: " << n / total_seconds << endl;
  
#ifdef DEBUG
  cout << "Device Filter Grad:";
  PrintTensor(filter_grad_on_device, c_out, c_in, kernel_h, kernel_w);
#endif

  // Do the host-side backward convolution w.r.t. the weights
  t1 = GetTime();
  ConvolutionBackwardFilterHost(
      images,
      n,
      c_in,
      h_in,
      w_in,
      output_grad,
      c_out,
      h_out,
      w_out,
      filter_grad,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w
      );

  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Host FPS: " << n / total_seconds << endl;

#ifdef DEBUG
  cout << "Filter Grad: ";
  PrintTensor(filter_grad, c_out, c_in, kernel_h, kernel_w);
#endif

  // Verify the results
  VerifyResults(filter_grad, filter_grad_on_device, c_out*c_in*kernel_h*kernel_w);

  // Do the device-side backward convolution w.r.t. the weights
  t1 = GetTime();
  ConvolutionBackwardFilter(
      images_on_device,
      n,
      c_in,
      h_in,
      w_in,
      output_grad_on_device,
      c_out,
      h_out,
      w_out,
      filter_grad_on_device,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      stream
      );

  CUDA_CALL(cudaStreamSynchronize(stream));
  total_seconds = ElapsedTime(t1, GetTime());
  cout << "Device FPS: " << n / total_seconds << endl;

  // Verify the results
  VerifyResults(filter_grad, filter_grad_on_device, c_out*c_in*kernel_h*kernel_w);

  // clean up
  delete[] filter_grad;
  delete[] images;
  delete[] output_grad;
  CUDA_CALL(cudaFree(filter_grad_on_device));
  CUDA_CALL(cudaFree(images_on_device));
  CUDA_CALL(cudaFree(output_grad_on_device));
  CUDA_CALL(cudaFree(workspace_on_device));
  CUDNN_CALL(cudnnDestroy(handle));
}
