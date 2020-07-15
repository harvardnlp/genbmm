// BANDED KERNELS
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include <curand.h>
#include <curand_kernel.h>
namespace {


template <typename scalar_t>
__global__ void banded_cuda_forward_kernel_mul(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const int n,
    const int a_lu, const int a_lb,
    const int b_lu, const int b_lb,
    const int result_lu, const int result_lb,
    const int mode
    ) {

  const int batch = blockIdx.z;
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Create outer dim
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
          // Loop over inner dim
          for (int k = 0; k < self_width; ++k) {
              pos = (i + (k - a_lu));
              k2 = (pos - o) + b_lu;
              if (k2 < 0 || k2 >= b_width) continue;
              if (pos < 0 || pos >= n) continue;
              scalar_t a_val = a[batch][i][k];
              scalar_t b_val = b[batch][o][k2];
              // done

              scalar_t v = a_val + b_val;
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
          maxes[batch][i][j] = m;
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
  const int o = i + (j - a_lu);

  if (i < n && j < a_lu + a_lb + 1 && o >= 0 && o < n) {

      scalar_t val = 0.0;
      const int gradout_width = result_lu + result_lb + 1;

      // Loop over outer (b) dimesion
      for (int k = 0; k < gradout_width; ++k) {
          const int pos = i + (k - result_lu);
          const int k2 = (o - pos) + b_lu;
          if (k2 < 0 || k2 >= b_lu + b_lb +1) continue;
          if (pos < 0 || pos >= n) continue;
          // END

          if (mode == 3) {
              val += b[batch][pos][k2] * grad_output[batch][i][k];

          } else if (mode == 1) {
              scalar_t v = (j == part[batch][i][k]) ? 1 : 0;
              val += v * grad_output[batch][i][k];

          } else if (mode == 0) {
              scalar_t v = a[batch][i][j] + b[batch][pos][k2] - part[batch][i][k];
              val += exp(v) * grad_output[batch][i][k];
          }
      }
      grad_a[batch][i][j] = val;
  }
}

template <typename scalar_t>
__global__ void banded_cuda_backbackward_kernel_A(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output_a,
    const int n,
    const int a_lu,
    const int a_lb,
    const int b_lu,
    const int b_lb,
    const int result_lu,
    const int result_lb) {

    // Left sided.
    const int batch = blockIdx.z;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int o = i + (j - a_lu);

    if (i < n && j < a_lu + a_lb + 1 && o >= 0 && o < n) {
        const int b_width = b_lu + b_lb + 1;
        scalar_t a_val = a[batch][i][j];
        // End Left sided.

        scalar_t val = 0.0;

        // Loop over right side.
        const int gradout_width = result_lu + result_lb + 1;
        for (int k = 0; k < gradout_width; ++k) {
            const int pos = i + (k - result_lu);
            const int k2 = (o - pos) + b_lu;
            if (k2 < 0 || k2 >= b_lu + b_lb +1) continue;
            if (pos < 0 || pos >= n) continue;
            scalar_t b_val = b[batch][pos][k2];
        // End over right side.

            scalar_t mx = maxes[batch][i][k];
            scalar_t z = exp(part[batch][i][k] -mx);
            scalar_t s = exp(a_val + b_val - mx) / z;
            scalar_t inner = 0.0;


            // Loop over inner dim
            const int self_width = a_lu + a_lb + 1;
            for (int m = 0; m < self_width; ++m) {
                const int pos_in = (i + (m - a_lu));
                int m2 = (pos_in - o) + b_lu;
                if (m2 < 0 || m2 >= b_width) continue;
                if (pos_in < 0 || pos_in >= n) continue;

                scalar_t a_inner_val = a[batch][i][m];
                scalar_t b_inner_val = b[batch][pos_in][m2];
                scalar_t s2 = exp(a_inner_val + b_inner_val - mx) / z;
                scalar_t v;
                if (j == m) {
                    v = s  - s * s2;
                } else {
                    v = - s * s2;
                }
                inner += v * grad_output_a[batch][i][m];
            }
            val += inner * grad_output[batch][i][k];
        }
        grad_a[batch][i][j] = val;
    }
}

template <typename scalar_t>
__global__ void banded_cuda_backbackward_kernel_B(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output_a,
    const int n,
    const int a_lu,
    const int a_lb,
    const int b_lu,
    const int b_lb,
    const int result_lu,
    const int result_lb) {

    // Right sided.
    const int batch = blockIdx.z;
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;
    const int k2 = threadIdx.y + blockIdx.y * blockDim.y;
    const int a_width = a_lu + a_lb + 1;
    const int b_width = b_lu + b_lb + 1;
    if (pos < n && k2 < b_lu + b_lb + 1) {
        /* const int o = i + (j - b_lu); */

        scalar_t b_val = b[batch][pos][k2];
        // End Right sided.

        scalar_t val = 0.0;

        // Loop over left side (not done).
        const int gradout_width = result_lu + result_lb + 1;
        for (int k = 0; k < gradout_width; ++k) {

            // fix these
            const int i = pos - (k - result_lu);
            const int j = k2 + pos - b_lu - (i - a_lu);
            if (j < 0 || j >= a_lu + a_lb +1) continue;
            if (i < 0 || i >= n) continue;
            //
            const int o =  i + (j - result_lu);
            scalar_t a_val = a[batch][i][j];
        // End over left side.

            scalar_t mx = maxes[batch][i][k];
            scalar_t z = exp(part[batch][i][k] - mx);
            scalar_t s = exp(a_val + b_val - mx) / z;
            scalar_t inner = 0.0;

            // Loop over inner dim
            const int self_width = a_lu + a_lb + 1;
            for (int m = 0; m < self_width; ++m) {
                const int pos_in = (i + (m - a_lu));
                int m2 = (pos_in - o) + b_lu;
                if (m2 < 0 || m2 >= b_width) continue;
                if (pos_in < 0 || pos_in >= n) continue;

                scalar_t a_inner_val = a[batch][i][m];
                scalar_t b_inner_val = b[batch][pos_in][m2];
                scalar_t s2 = exp(a_inner_val + b_inner_val - mx) / z;
                scalar_t v;
                if (j == m) {
                    v = s  - s * s2;
                } else {
                    v = - s * s2;
                }
                inner += v * grad_output_a[batch][i][m];
            }
            val += inner * grad_output[batch][i][k];
        }
        grad_b[batch][pos][k2] = val;
    }

}


template <typename scalar_t>
__global__ void banded_cuda_backbackward_kernel_C(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_grad,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> part,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> maxes,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output_a,
    const int n,
    const int a_lu,
    const int a_lb,
    const int b_lu,
    const int b_lb,
    const int result_lu,
    const int result_lb) {

    // Full size
    const int batch = blockIdx.z;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int o =  i + (j - result_lu);

    if (i < n && j < result_lu + result_lb + 1 && o >= 0 && o < n) {
        const int self_width = a_lu + a_lb + 1;
        const int b_width = b_lu + b_lb + 1;

        int k2 = 0;
        int pos = 0;
        if (o < 0 || o >= n) return;
        scalar_t val = 0.0;

        scalar_t mx = maxes[batch][i][j];
        // Loop over inner dim
        for (int k = 0; k < self_width; ++k) {
            pos = (i + (k - a_lu));
            k2 = (pos - o) + b_lu;
            if (k2 < 0 || k2 >= b_width) continue;
            if (pos < 0 || pos >= n) continue;
            scalar_t a_val = a[batch][i][k];
            scalar_t b_val = b[batch][o][k2];
            // done

            val += (exp(a_val + b_val - mx) / (exp(part[batch][i][j] - mx))) * grad_output_a[batch][i][k];
        }
        grad_grad[batch][i][j] = val;
    }
}
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
    auto maxes = torch::zeros({batch_size, a_size, new_size}, options);


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
                    maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    a_size, a_lu, a_lb, b_lu, b_lb,
                    out_lu, out_lb,
                    mode);

            } ) );
    return {out, indices, maxes};
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


std::vector<torch::Tensor> banded_cuda_backbackward(
        torch::Tensor a,
        int a_lu,
        int a_lb,
        torch::Tensor b,
        int b_lu,
        int b_lb,
        torch::Tensor grad_output,
        torch::Tensor part,
        torch::Tensor maxes,
        torch::Tensor grad_output_a,
        int mode) {

    const int batch_size = a.size(0);
    const int out_lu = a_lu + b_lb;
    const int out_lb = a_lb + b_lu;

    const int a_size = a.size(1);
    const int b_size = b.size(1);
    const int new_size = out_lu + out_lb + 1;

    auto grad_a = torch::zeros_like(a);
    {
    const int in_size = a.size(2);
    const int threads = 16;
    const dim3 blocks(a_size / threads + 1,
                      in_size / threads + 1,
                      batch_size);
    const dim3 threads_per_block(threads, threads, 1);


    AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
       banded_cuda_backbackward_kernel_A<scalar_t><<<blocks, threads_per_block>>>(
           grad_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a_size, a_lu, a_lb, b_lu, b_lb,
           out_lu, out_lb);
            }));
    }

    auto grad_b = torch::zeros_like(b);
    {
    const int in_size = b.size(2);
    const int threads = 16;
    const dim3 blocks(b_size / threads + 1,
                      in_size / threads + 1,
                      batch_size);
    const dim3 threads_per_block(threads, threads, 1);


    AT_DISPATCH_FLOATING_TYPES(b.type(), "matmul_forward_cuda", ([&] {
       banded_cuda_backbackward_kernel_B<scalar_t><<<blocks, threads_per_block>>>(
           grad_b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a_size, a_lu, a_lb, b_lu, b_lb,
           out_lu, out_lb);
            }));
    }

    auto grad_grad = torch::zeros_like(grad_output);
    {
    const int threads = 16;
    const dim3 blocks(a_size / threads + 1,
                      new_size / threads + 1,
                      batch_size);
    const dim3 threads_per_block(threads, threads, 1);

    AT_DISPATCH_FLOATING_TYPES(a.type(), "matmul_forward_cuda", ([&] {
       banded_cuda_backbackward_kernel_C<scalar_t><<<blocks, threads_per_block>>>(
           grad_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           b.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           part.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           maxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           grad_output_a.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
           a_size, a_lu, a_lb, b_lu, b_lb,
           out_lu, out_lb

                                                                              );
            }));
    }
    return {grad_a, grad_b, grad_grad};

}
