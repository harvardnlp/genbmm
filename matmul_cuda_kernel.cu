#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>
namespace {

template <typename scalar_t>
__global__ void matmul_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  if (row < a_size && col < b_size) {
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
         }
      }
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         val += exp(v - m);
      }
      out[n][row][col] = log(val) + m;
  }
}

template <typename scalar_t>
__global__ void max_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  int ind = -1;
  if (row < a_size && col < b_size) {
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
             ind = i;
         }
      }
      out[n][row][col] = m;
      indices[n][row][col] = ind;
  }
}

template <typename scalar_t>
__global__ void sample_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> rand,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  scalar_t val = 0.0;
  scalar_t m = -1e9;
  int ind = -1;
  if (row < a_size && col < b_size) {

      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         if (v > m) {
             m = v;
         }
      }
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col];
         val += exp(v - m);
      }
      out[n][row][col] = log(val) + m;

      scalar_t total = 0.0;
      auto r = rand[n][row][col];
      for (int i = 0; i < in_size; ++i) {
         scalar_t v = a[n][row][i] + b[n][i][col] - out[n][row][col];
         if (total < r && total + exp(v) > r ){
             indices[n][row][col] = i;
             break;
         }
         total += exp(v);
      }

  }
}


// BACKWARD

// LOGSUM

template <typename scalar_t>
__global__ void matmul_cuda_backward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < b_size; ++k) {
         scalar_t v = a[n][row][col] + b[n][col][k] - part[n][row][k];
         val += exp(v) * grad_output[n][row][k];
      }
      grad_a[n][row][col] = val;
  }
}
template <typename scalar_t>
__global__ void matmul_cuda_backward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < a_size; ++k) {
         scalar_t v = a[n][k][row] + b[n][row][col] - part[n][k][col];
         val += exp(v) * grad_output[n][k][col];
      }
      grad_b[n][row][col] = val;
  }
}

// MAX

template <typename scalar_t>
__global__ void max_cuda_backward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < a_size && col < in_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < b_size; ++k) {
          scalar_t v = (col == part[n][row][k]) ? 1 : 0;
          val += v * grad_output[n][row][k];
      }
      grad_a[n][row][col] = val;
  }
}

template <typename scalar_t>
__global__ void max_cuda_backward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int in_size,
    const int a_size,
    const int b_size
    ) {

  const int n = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < in_size && col < b_size) {
      scalar_t val = 0.0;
      for (int k = 0; k < a_size; ++k) {
          scalar_t v = (row == part[n][k][col]) ? 1 : 0;
          val += v * grad_output[n][k][col];
      }
      grad_b[n][row][col] = val;
  }
}



} // namespace



std::vector<torch::Tensor> matmul_cuda_forward(
    torch::Tensor a,
    torch::Tensor b,
    int mode) {

  const int batch_size = a.size(0);
  const int a_size = a.size(1);
  const int b_size = b.size(2);

  auto options = torch::TensorOptions()
          .dtype(a.dtype())
          .device(torch::kCUDA, a.device().index());
  auto out = torch::zeros({batch_size, a_size, b_size}, options);

  const int in_size = a.size(2);
  const int threads = 32;
  const dim3 threads_per_block(threads, threads, 1);
  const dim3 blocks(a_size / threads + 1,
                    b_size / threads + 1,
                    batch_size);

  // Dispatch
  if (mode == 0) {
      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  matmul_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
        return {out};
  } else if (mode == 1) {
      auto options2 = torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCUDA, a.device().index());


      auto indices = torch::zeros({batch_size, a_size, b_size}, options2);
      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, indices};
  } else if (mode == 2) {
      auto options2 = torch::TensorOptions()
              .dtype(torch::kInt)
              .device(torch::kCUDA, a.device().index());
      auto indices = torch::zeros({batch_size, a_size, b_size}, options2);
      auto rand = torch::rand({batch_size, a_size, b_size}, options);
      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  sample_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      rand.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              } ) );
      return {out, indices};
  }

}

std::vector<torch::Tensor> matmul_cuda_backward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_out,
    torch::Tensor part,
    int mode) {

  const auto batch_size = a.size(0);
  const auto in_size = a.size(2);
  const int a_size = a.size(1);
  const int b_size = b.size(2);

  const int threads = 32;
  const dim3 blocks(a_size / threads + 1,
                    in_size / threads + 1,
                    batch_size);
  const dim3 threads_per_block(threads, threads, 1);
  auto grad_a = torch::zeros_like(a);


  auto grad_b = torch::zeros_like(b);
  auto grad_bp = grad_b.packed_accessor32<float,3,torch::RestrictPtrTraits>();
  const int threads2 = 32;
  const dim3 blocks2(in_size / threads2 + 1,
                    b_size / threads2 + 1,
                    batch_size);

  if (mode == 0) {
      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  matmul_cuda_backward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size
                                                                                         );
              }));

      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  matmul_cuda_backward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>(
                      grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));
  } else if (mode == 1 or mode == 2) {

      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_backward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
                      grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));

      AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
                  max_cuda_backward_kernel_B<scalar_t><<<blocks2, threads_per_block>>>(
                      grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                      in_size, a_size, b_size);
              }));
  }
  return {grad_a, grad_b};
}
