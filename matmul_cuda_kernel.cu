#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>


#define TPB 32

namespace {

// FORWARD KERNELS

template <typename scalar_t>
__global__ void matmul_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    const int in_size,
    const int a_size,
    const int b_size) {

  __shared__ scalar_t sA[TPB * TPB];
  __shared__ scalar_t sB[TPB * TPB];

  const int batch = blockIdx.z;
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  const int col = threadIdx.y + blockIdx.y * blockDim.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int bpg = blockDim.x;

  if (row >= a_size && col >= b_size)
      return;


  scalar_t m = -1e9;
  __syncthreads();

  for (int q = 0; q < bpg; q++) {
      sA[tx * TPB + ty] = a[batch][row][ty + q * TPB];
      sB[tx * TPB + ty] = b[batch][tx + q * TPB][col];

      __syncthreads();
      for (int i = 0; i < TPB; ++i) {
          scalar_t v = sA[tx * TPB + i] + sB[i * TPB + ty];
          if (v > m)
              m = v;
      }
      __syncthreads();
  }
  scalar_t val = 0.0;
  for (int q = 0; q < bpg; q++) {

      sA[tx * TPB + ty] = a[batch][row][ty + q * TPB];
      sB[tx * TPB + ty] = b[batch][tx + q * TPB][col];
      __syncthreads();

      for (int i = 0; i < TPB; ++i) {
          scalar_t v = sA[tx * TPB + i] + sB[i * TPB + ty];
          val += exp(v - m);
      }
      __syncthreads();
  }

  out[batch][row][col] = log(val) + m;

  return;
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


// BACKWARD KERNELS

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

// MAX / SAMPLE

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



// BANDED KERNELS


template <typename scalar_t>
__global__ void banded_cuda_forward_kernel_mul(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    const int n,
    const int a_lu,
    const int a_lb,
    const int b_lu,
    const int b_lb,
    const int result_lu,
    const int result_lb,
    const int mode
    ) {

  const int batch = blockIdx.z;
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < n && j < result_lu + result_lb + 1) {
      const int self_width = a_lu + a_lb + 1;
      const int b_width = b_lu + b_lb + 1;
      const int o =  i + (j - result_lu);
      int k2 = 0;
      int pos = 0;
      if (o < 0 || o >= n) return;

      if (mode == 1) {
          scalar_t val = 0.0;
          scalar_t m = -1e9;
          int ind = -1;
          for (int k = 0; k < self_width; ++k) {
              pos = (i + (k - a_lu));
              k2 = (pos - o) + b_lu;
              if (k2 < 0 || k2 >= b_width) continue;
              if (pos < 0 || pos >= n) continue;

              scalar_t v = a[batch][i][k] + b[batch][o][k2];
              if (v > m) {
                  m = v;
                  ind = k;
              }
          }
          out[batch][i][j] = m;
          indices[batch][i][j] = ind;

      } else if (mode == 3) {
          scalar_t val = 0.0;
          for (int k = 0; k < self_width; ++k) {
              pos = (i + (k - a_lu));
              k2 = (pos - o) + b_lu;
              if (k2 < 0 || k2 >= b_width) continue;
              if (pos < 0 || pos >= n) continue;

              val += a[batch][i][k] * b[batch][o][k2];
          }
          out[batch][i][j] = val;
      } else if (mode == 0) {

          scalar_t val = 0.0;
          scalar_t m = -1e9;
          for (int k = 0; k < self_width; ++k) {
              pos = (i + (k - a_lu));
              if (pos < 0 || pos >= n) continue;
              k2 = (pos - o) + b_lu;
              if (k2 < 0 || k2 >= b_width) continue;

              scalar_t v = a[batch][i][k] + b[batch][o][k2];
              if (v > m) m = v;
          }
          for (int k = 0; k < self_width; ++k) {
              pos = (i + (k - a_lu));
              if (pos < 0 || pos >= n) continue;
              k2 = (pos - o) + b_lu;
              if (k2 < 0 || k2 >= b_width) continue;
              val += exp(a[batch][i][k] + b[batch][o][k2] - m);
          }
          out[batch][i][j] = log(val) + m;
      }
  }
}



template <typename scalar_t>
__global__ void banded_cuda_backward_kernel_mul(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const int n,
    const int a_lu,
    const int a_lb,
    const int b_lu,
    const int b_lb,
    const int result_lu,
    const int result_lb,
    const int mode) {

  const int batch = blockIdx.z;
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < n && j < a_lu + a_lb + 1) {
      const int o = i + (j - a_lu);
      scalar_t val = 0.0;
      const int gradout_width = result_lu + result_lb + 1;

      if (mode == 3) {
          for (int k = 0; k < gradout_width; ++k) {
              const int pos = i + (k - result_lu);
              const int k2 = (o - pos) + b_lu;
              if (k2 < 0 || k2 >= b_lu + b_lb +1) continue;
              if (pos < 0 || pos >= n) continue;
              val += b[batch][pos][k2] * grad_output[batch][i][k];
          }
      } else if (mode == 1) {
          // Max
          for (int k = 0; k < gradout_width; ++k) {
              const int pos = i + (k - result_lu);
              const int k2 = (o - pos) + b_lu;
              if (k2 < 0 || k2 >= b_lu + b_lb +1) continue;
              if (pos < 0 || pos >= n) continue;

              scalar_t v = (j == part[batch][i][k]) ? 1 : 0;
              val += v * grad_output[batch][i][k];
          }

      } else if (mode == 0) {
          for (int k = 0; k < gradout_width; ++k) {
              const int pos = i + (k - result_lu);
              if (pos < 0 || pos >= n) continue;
              const int k2 = (o - pos) + b_lu;
              if (k2 < 0 || k2 >= b_lu + b_lb +1) continue;

              scalar_t v = a[batch][i][j] + b[batch][pos][k2] - part[batch][i][k];
              val += exp(v) * grad_output[batch][i][k];
          }
      }
      grad_a[batch][i][j] = val;
  }
}

} // namespace


// MATMUL FORWARD DISPATCH


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

// MATMUL BACKWARD DISPATCH
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

// BANDED FORWARD
std::vector<torch::Tensor> banded_cuda_forward(
    torch::Tensor a,
    int a_lu,
    int a_lb,
    torch::Tensor b,
    int b_lu,
    int b_lb,
    int mode) {

    const int batch_size = a.size(0);
    const int out_lu = a_lu + b_lb;
    const int out_lb = a_lb + b_lu;

    const int a_size = a.size(1);
    const int new_size = out_lu + out_lb + 1;

    auto options = torch::TensorOptions()
            .dtype(a.dtype())
            .device(torch::kCUDA, a.device().index());
    auto out = torch::zeros({batch_size, a_size, new_size}, options);

    const int in_size = a.size(2);
    const int threads = 32;
    const dim3 threads_per_block(threads, threads, 1);
    const dim3 blocks(a_size / threads + 1,
                      new_size / threads + 1,
                      batch_size);

    auto options2 = torch::TensorOptions()
            .dtype(torch::kInt)
            .device(torch::kCUDA, a.device().index());
    auto indices = torch::zeros({batch_size, a_size, new_size}, options2);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "banded_forward_cuda", ([&] {
                banded_cuda_forward_kernel_mul<scalar_t><<<blocks, threads_per_block>>>(
                    a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    indices.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                    a_size, a_lu, a_lb, b_lu, b_lb,
                    out_lu, out_lb,
                    mode);

            } ) );
    return {out, indices};



}

std::vector<torch::Tensor> banded_cuda_backward(
        torch::Tensor a,
        int a_lu,
        int a_lb,
        torch::Tensor b,
        int b_lu,
        int b_lb,
        torch::Tensor grad_output,
        torch::Tensor part,
        int mode) {

    const int batch_size = a.size(0);
    const int out_lu = a_lu + b_lb;
    const int out_lb = a_lb + b_lu;

    const int a_size = a.size(1);
    const int new_size = out_lu + out_lb + 1;

    auto options = torch::TensorOptions()
            .dtype(a.dtype())
            .device(torch::kCUDA, a.device().index());
    auto out = torch::zeros({batch_size, a_size, new_size}, options);

    const int in_size = a.size(2);
    const int threads = 32;
    const dim3 blocks(a_size / threads + 1,
                      in_size / threads + 1,
                      batch_size);
    const dim3 threads_per_block(threads, threads, 1);
    auto grad_a = torch::zeros_like(a);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
       banded_cuda_backward_kernel_mul<scalar_t><<<blocks, threads_per_block>>>(
           grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a_size, a_lu, a_lb, b_lu, b_lb,
           out_lu, out_lb,
           mode

                                                                              );
            }));
    return {grad_a};

}
