#include "type_convert.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"
#include "core/batch_invariant.hpp"
#include "quantization/vectorization_utils.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {

// Warp-level reduction for better performance on smaller hidden sizes
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Optimized RMS norm kernel with vectorized normalization pass
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  __shared__ float warp_variances[32];  // Max 32 warps per block

  float variance = 0.0f;
  const scalar_t* input_row = input + blockIdx.x * input_stride;
  scalar_t* out_row = out + blockIdx.x * hidden_size;

  // First pass: compute variance with vectorized reads
  constexpr int VEC_SIZE = 8;
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  // Warp-level reduction followed by block-level reduction
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  variance = warpReduceSum(variance);

  if (lane_id == 0) {
    warp_variances[warp_id] = variance;
  }
  __syncthreads();

  // Final reduction across warps
  if (threadIdx.x < 32) {
    variance = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_variances[threadIdx.x] : 0.0f;
    variance = warpReduceSum(variance);
  }

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Second pass: apply normalization with vectorized reads/writes
  const float rms_scale = s_variance;

  // Vectorized normalization for aligned accesses
  auto norm_vec_op = [&](const vec_n_t<scalar_t, VEC_SIZE>& in_vec,
                         const vec_n_t<scalar_t, VEC_SIZE>& weight_vec,
                         int idx) {
    vec_n_t<scalar_t, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(in_vec.val[i]);
      float w = static_cast<float>(weight_vec.val[i]);
      out_vec.val[i] = static_cast<scalar_t>((x * rms_scale) * w);
    }
    *reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(&out_row[idx]) = out_vec;
  };

  auto norm_scalar_op = [&](const scalar_t& in_val, const scalar_t& weight_val, int idx) {
    float x = static_cast<float>(in_val);
    float w = static_cast<float>(weight_val);
    out_row[idx] = static_cast<scalar_t>((x * rms_scale) * w);
  };

  // Process elements with vectorization where possible
  const int num_vec_elems = (hidden_size / VEC_SIZE) * VEC_SIZE;
  for (int idx = threadIdx.x * VEC_SIZE; idx < num_vec_elems; idx += blockDim.x * VEC_SIZE) {
    vec_n_t<scalar_t, VEC_SIZE> in_vec =
        *reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(&input_row[idx]);
    vec_n_t<scalar_t, VEC_SIZE> weight_vec =
        *reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(&weight[idx]);
    norm_vec_op(in_vec, weight_vec, idx);
  }

  // Handle remaining elements
  for (int idx = num_vec_elems + threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    norm_scalar_op(input_row[idx], weight[idx], idx);
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  __shared__ float warp_variances[32];  // Max 32 warps per block

  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  // First pass: add and compute variance with loop unrolling
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  // Warp-level reduction followed by block-level reduction
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  variance = warpReduceSum(variance);

  if (lane_id == 0) {
    warp_variances[warp_id] = variance;
  }
  __syncthreads();

  // Final reduction across warps
  if (threadIdx.x < 32) {
    variance = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_variances[threadIdx.x] : 0.0f;
    variance = warpReduceSum(variance);
  }

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Second pass: apply normalization with loop unrolling
  const float rms_scale = s_variance;
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= rms_scale;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  __shared__ float warp_variances[32];  // Max 32 warps per block

  float variance = 0.0f;

  // First pass: add and compute variance with loop unrolling
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  // Warp-level reduction followed by block-level reduction
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  variance = warpReduceSum(variance);

  if (lane_id == 0) {
    warp_variances[warp_id] = variance;
  }
  __syncthreads();

  // Final reduction across warps
  if (threadIdx.x < 32) {
    variance = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_variances[threadIdx.x] : 0.0f;
    variance = warpReduceSum(variance);
  }

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Second pass: apply normalization with loop unrolling
  const float rms_scale = s_variance;
  #pragma unroll 4
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    input[blockIdx.x * input_stride + idx] = (scalar_t)((x * rms_scale) * w);
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);

  // We cannot just use `input.stride(-2)` if the tensor is not row-major.
  // Instead, we use a 2d view to get the second-innermost stride.
  // That way the dimensions (except the last one) can be arbitrarily permuted.
  torch::Tensor input_view = input.view({-1, hidden_size});

  int num_tokens = input_view.numel() / hidden_size;
  int64_t input_stride = input_view.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_view));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input_view.scalar_type(), "rms_norm_kernel", [&] {
        vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), input_view.data_ptr<scalar_t>(),
            input_stride, weight.data_ptr<scalar_t>(), epsilon, num_tokens,
            hidden_size);
      });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                    \
  VLLM_DISPATCH_FLOATING_TYPES(                                             \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {               \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                    \
            <<<grid, block, 0, stream>>>(                                   \
                input.data_ptr<scalar_t>(), input_stride,                   \
                residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), \
                epsilon, num_tokens, hidden_size);                          \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      !batch_invariant_launch) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
