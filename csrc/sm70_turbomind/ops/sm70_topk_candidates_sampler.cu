// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Small-batch top-k/top-p sampling from vocab-parallel candidates (SM70).
//
// 1. sm70_local_top32_half_wire: each TP rank selects the 32 largest of its
//    FP16 logits shard and writes them as one FP32 row
//    [32 values (descending), 32 global token ids as exact floats].
// 2. The caller all-gathers these rows across TP ranks (rank-major, dim 0).
// 3. sm70_sample_topk_candidates_out: per request, the vLLM reference
//    top-k/top-p sampler (apply_top_k_top_p_pytorch + random_sample)
//    restricted to the gathered candidates:
//      - temperature division in FP32 (round-to-nearest, as logits.div_),
//      - top-k keeps every candidate equal to the k-th value (the reference
//        masks only values strictly below it),
//      - top-p walks the kept candidates in ascending order (ties: lower
//        token id first, as the reference's stable ascending sort) and masks
//        while the cumulative probability is <= 1 - p, never the largest,
//      - argmax(prob / q) where q is the exact exponential the reference's
//        q.exponential_() would draw for that token from the same Philox
//        generator state; the generator advances by the same offset.
//    Top-k never needs more than k candidates, so with k <= 20 and 32 per
//    rank the result matches the full-vocabulary reference except when more
//    than 12 extra logits tie the k-th value on one rank.

#include <torch/all.h>

#include <ATen/core/TransformationHelper.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>

namespace {

constexpr int kLocalK = 32;
constexpr int kWireWidth = 2 * kLocalK;
constexpr int kSelectThreads = 256;
constexpr int kTileItems = 8;
constexpr int kTile = kSelectThreads * kTileItems;  // 2048 logits per block
constexpr int kMaxTiles = 64;                       // shard <= 131072 logits
constexpr int kMergeItems = kMaxTiles * kLocalK / kSelectThreads;  // 8
constexpr int kMaxCandidates = 128;                 // TP <= 4
constexpr int kMaxRows = 8;
constexpr int kNativeBlockSize = 256;
constexpr int kNativeUnroll = 4;
constexpr int kMaxGeneratorOffsetsPerCurandCall = 4;

static_assert(kMergeItems * kSelectThreads >= kMaxTiles * kLocalK);

// Order-preserving 16-bit key of an FP16 value; 0 is reserved for "empty".
// -0 sorts as +0 and NaN sorts below -inf.
__device__ __forceinline__ uint32_t half_order(uint32_t bits) {
  if ((bits & 0x7fffu) == 0u) {
    bits = 0u;
  }
  if ((bits & 0x7fffu) > 0x7c00u) {
    return 1u;
  }
  return (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u);
}

// Stage 1: per 2048-logit tile, the 32 largest as (order << 16 | ~local).
__global__ void __launch_bounds__(kSelectThreads)
    local_top32_tile_kernel(const uint16_t* __restrict__ logits,
                            int64_t row_stride, int num_cols, int num_valid,
                            int tiles, bool vector_loads,
                            uint32_t* __restrict__ partial) {
  using Sort = cub::BlockRadixSort<uint32_t, kSelectThreads, kTileItems>;
  __shared__ typename Sort::TempStorage storage;

  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  const uint16_t* src = logits + row * row_stride;
  const int local_base = threadIdx.x * kTileItems;
  const int base = tile * kTile + local_base;

  uint16_t bits[kTileItems];
  if (vector_loads && base + kTileItems <= num_cols) {
    const uint4 packed = *reinterpret_cast<const uint4*>(src + base);
    const uint16_t* lanes = reinterpret_cast<const uint16_t*>(&packed);
#pragma unroll
    for (int i = 0; i < kTileItems; ++i) {
      bits[i] = lanes[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < kTileItems; ++i) {
      bits[i] = base + i < num_cols ? src[base + i] : uint16_t(0);
    }
  }

  uint32_t keys[kTileItems];
#pragma unroll
  for (int i = 0; i < kTileItems; ++i) {
    keys[i] = base + i < num_valid
                  ? (half_order(bits[i]) << 16) |
                        (0xffffu - static_cast<uint32_t>(local_base + i))
                  : 0u;
  }
  // Only the value half needs ordering; the tile offset rides along.
  Sort(storage).SortDescending(keys, 16, 32);

  uint32_t* out = partial + (static_cast<int64_t>(row) * tiles + tile) * kLocalK;
#pragma unroll
  for (int i = 0; i < kTileItems; ++i) {
    const int rank = local_base + i;
    if (rank < kLocalK) {
      out[rank] = keys[i];
    }
  }
}

// Stage 2: per row, the 32 largest of the tiles' candidates -> wire row.
__global__ void __launch_bounds__(kSelectThreads)
    local_top32_merge_kernel(const uint16_t* __restrict__ logits,
                             int64_t row_stride, int tiles,
                             int64_t vocab_start,
                             const uint32_t* __restrict__ partial,
                             float* __restrict__ wire) {
  using Sort = cub::BlockRadixSort<uint32_t, kSelectThreads, kMergeItems>;
  __shared__ typename Sort::TempStorage storage;

  const int row = blockIdx.x;
  const int count = tiles * kLocalK;
  const uint32_t* row_partial = partial + static_cast<int64_t>(row) * count;

  uint32_t keys[kMergeItems];
#pragma unroll
  for (int i = 0; i < kMergeItems; ++i) {
    const int position = threadIdx.x * kMergeItems + i;
    const uint32_t key = position < count ? row_partial[position] : 0u;
    keys[i] = key == 0u ? 0u
                        : (key & 0xffff0000u) |
                              (0xffffu - static_cast<uint32_t>(position));
  }
  Sort(storage).SortDescending(keys, 16, 32);

  float* out = wire + static_cast<int64_t>(row) * kWireWidth;
#pragma unroll
  for (int i = 0; i < kMergeItems; ++i) {
    const int rank = threadIdx.x * kMergeItems + i;
    if (rank < kLocalK) {
      float value = -INFINITY;
      float token = -1.f;
      if (keys[i] != 0u) {
        const int position = static_cast<int>(0xffffu - (keys[i] & 0xffffu));
        const int local = static_cast<int>(
            0xffffu - (row_partial[position] & 0xffffu));
        const int index = (position / kLocalK) * kTile + local;
        value = __half2float(
            __ushort_as_half(logits[row * row_stride + index]));
        token = static_cast<float>(vocab_start + index);
      }
      out[rank] = value;
      out[kLocalK + rank] = token;
    }
  }
}

// The exponential that at::native exponential_ (float, CUDA) writes at
// linear `index` of a tensor whose launch used `total_threads` threads.
__device__ __forceinline__ float native_exponential_for_index(
    int64_t index, int64_t total_threads, at::PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  const int64_t values_per_round = total_threads * kNativeUnroll;
  const int64_t round = index / values_per_round;
  const int64_t within_round = index - round * values_per_round;
  const uint64_t subsequence =
      static_cast<uint64_t>(within_round % total_threads);
  const int component = static_cast<int>(within_round / total_threads);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, subsequence, offset, &state);
  float4 random = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int64_t iteration = 0; iteration <= round; ++iteration) {
    random = curand_uniform4(&state);
  }
  const float uniform = reinterpret_cast<const float*>(&random)[component];
  return at::transformation::exponential<float>(uniform, 1.f);
}

__global__ void __launch_bounds__(kMaxCandidates)
    sample_topk_candidates_kernel(
        int32_t* __restrict__ sampled, const float* __restrict__ wire,
        const float* __restrict__ temperature,
        const int32_t* __restrict__ top_k, const float* __restrict__ top_p,
        int rows, int tp_size, int64_t vocab_size, int64_t row_index_stride,
        int64_t total_threads, at::PhiloxCudaState philox_args) {
  __shared__ float s_value[kMaxCandidates];
  __shared__ int s_token[kMaxCandidates];
  __shared__ float s_sorted_value[kMaxCandidates];
  __shared__ int s_sorted_token[kMaxCandidates];
  __shared__ float s_weight[kMaxCandidates];
  __shared__ float s_exponential[kMaxCandidates];

  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const int candidates = tp_size * kLocalK;

  bool valid = false;
  float value = -INFINITY;
  int token = -1;
  if (t < candidates) {
    const float* src =
        wire + (static_cast<int64_t>(t / kLocalK) * rows + row) * kWireWidth;
    const float raw = src[t % kLocalK];
    const float id = src[kLocalK + t % kLocalK];
    if (id >= 0.f && id < static_cast<float>(vocab_size) && !isnan(raw)) {
      // logits.div_(temperature): IEEE division even under fast math.
      value = __fdiv_rn(raw, temperature[row]);
      token = static_cast<int>(id);
      valid = !isnan(value);
    }
  }
  s_value[t] = value;
  s_token[t] = valid ? token : -1;
  const int valid_count = __syncthreads_count(valid);

  // Descending rank. Equal values order by descending token id: the exact
  // reverse of the reference's stable ascending sort.
  if (valid) {
    int rank = 0;
    for (int s = 0; s < candidates; ++s) {
      const int other = s_token[s];
      const float other_value = s_value[s];
      if (other >= 0 &&
          (other_value > value || (other_value == value && other > token))) {
        ++rank;
      }
    }
    s_sorted_value[rank] = value;
    s_sorted_token[rank] = token;
  }
  __syncthreads();
  if (valid_count == 0) {
    if (t == 0) {
      sampled[row] = 0;
    }
    return;
  }

  int k = top_k[row];
  k = k < 1 ? 1 : (k > valid_count ? valid_count : k);
  const float kth_value = s_sorted_value[k - 1];
  const float max_value = s_sorted_value[0];
  if (t < valid_count) {
    // Rank t is kept by top-k iff it is not below the k-th value.
    s_weight[t] = s_sorted_value[t] >= kth_value
                      ? expf(s_sorted_value[t] - max_value)
                      : -1.f;
    s_exponential[t] = native_exponential_for_index(
        row * row_index_stride + s_sorted_token[t], total_threads,
        philox_args);
  }
  __syncthreads();

  if (t == 0) {
    int kept = 0;
    float total = 0.f;
    while (kept < valid_count && s_weight[kept] >= 0.f) {
      total += s_weight[kept];
      ++kept;
    }
    if (top_p != nullptr) {
      const float threshold = 1.f - top_p[row];
      float cumulative = 0.f;
      for (int i = kept - 1; i >= 1; --i) {
        cumulative += s_weight[i] / total;
        if (cumulative <= threshold) {
          s_weight[i] = 0.f;
        }
      }
    }
    float renormalized = 0.f;
    for (int i = 0; i < kept; ++i) {
      renormalized += s_weight[i];
    }
    float best_score = -1.f;
    int best_token = s_sorted_token[0];
    for (int i = 0; i < kept; ++i) {
      const float score = (s_weight[i] / renormalized) / s_exponential[i];
      const int candidate = s_sorted_token[i];
      // torch.argmax keeps the first (lowest token id) maximum.
      if (score > best_score ||
          (score == best_score && candidate < best_token)) {
        best_score = score;
        best_token = candidate;
      }
    }
    sampled[row] = best_token;
  }
}

}  // namespace

void sm70_local_top32_half_wire(torch::Tensor logits, torch::Tensor partial,
                                torch::Tensor wire, int64_t vocab_start,
                                int64_t num_valid) {
  TORCH_CHECK(logits.is_cuda() && partial.is_cuda() && wire.is_cuda(),
              "sm70_local_top32_half_wire: tensors must be CUDA");
  TORCH_CHECK(partial.device() == logits.device() &&
                  wire.device() == logits.device(),
              "sm70_local_top32_half_wire: device mismatch");
  TORCH_CHECK(logits.scalar_type() == torch::kHalf &&
                  partial.scalar_type() == torch::kInt32 &&
                  wire.scalar_type() == torch::kFloat32,
              "sm70_local_top32_half_wire: dtype mismatch");
  TORCH_CHECK(logits.dim() == 2 && logits.stride(1) == 1,
              "sm70_local_top32_half_wire: logits must be row-major 2D");
  const int64_t rows = logits.size(0);
  const int64_t cols = logits.size(1);
  const int64_t tiles = (cols + kTile - 1) / kTile;
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows,
              "sm70_local_top32_half_wire: unsupported row count");
  TORCH_CHECK(tiles <= kMaxTiles && num_valid >= kLocalK && num_valid <= cols,
              "sm70_local_top32_half_wire: unsupported shard width");
  TORCH_CHECK(vocab_start >= 0 && vocab_start + cols <= (int64_t(1) << 24),
              "sm70_local_top32_half_wire: token ids must fit FP32 exactly");
  TORCH_CHECK(logits.stride(0) >= cols &&
                  logits.stride(0) <= (int64_t(1) << 31) / rows,
              "sm70_local_top32_half_wire: unsupported row stride");
  TORCH_CHECK(partial.is_contiguous() && partial.numel() == rows * tiles *
                                                               kLocalK,
              "sm70_local_top32_half_wire: partial shape mismatch");
  TORCH_CHECK(wire.is_contiguous() && wire.dim() == 2 && wire.size(0) == rows &&
                  wire.size(1) == kWireWidth,
              "sm70_local_top32_half_wire: wire shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const auto* data = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  const bool vector_loads =
      reinterpret_cast<uintptr_t>(data) % 16 == 0 && logits.stride(0) % 8 == 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  local_top32_tile_kernel<<<dim3(tiles, rows), kSelectThreads, 0, stream>>>(
      data, logits.stride(0), static_cast<int>(cols),
      static_cast<int>(num_valid), static_cast<int>(tiles), vector_loads,
      reinterpret_cast<uint32_t*>(partial.data_ptr<int32_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  local_top32_merge_kernel<<<rows, kSelectThreads, 0, stream>>>(
      data, logits.stride(0), static_cast<int>(tiles), vocab_start,
      reinterpret_cast<const uint32_t*>(partial.data_ptr<int32_t>()),
      wire.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_sample_topk_candidates_out(
    torch::Tensor sampled, torch::Tensor wire, torch::Tensor temperature,
    torch::Tensor top_k, const std::optional<torch::Tensor>& top_p,
    const std::optional<at::Generator>& generator, int64_t vocab_size,
    int64_t tp_size) {
  TORCH_CHECK(sampled.is_cuda() && wire.is_cuda() && temperature.is_cuda() &&
                  top_k.is_cuda(),
              "sm70_sample_topk_candidates_out: tensors must be CUDA");
  TORCH_CHECK(sampled.scalar_type() == torch::kInt32 &&
                  wire.scalar_type() == torch::kFloat32 &&
                  temperature.scalar_type() == torch::kFloat32 &&
                  top_k.scalar_type() == torch::kInt32,
              "sm70_sample_topk_candidates_out: dtype mismatch");
  TORCH_CHECK(tp_size >= 1 && tp_size * kLocalK <= kMaxCandidates,
              "sm70_sample_topk_candidates_out: unsupported TP size");
  TORCH_CHECK(wire.dim() == 2 && wire.size(1) == kWireWidth &&
                  wire.size(0) % tp_size == 0 && wire.is_contiguous(),
              "sm70_sample_topk_candidates_out: wire shape mismatch");
  const int64_t rows = wire.size(0) / tp_size;
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows,
              "sm70_sample_topk_candidates_out: unsupported row count");
  TORCH_CHECK(sampled.numel() == rows && sampled.is_contiguous() &&
                  temperature.numel() == rows && temperature.is_contiguous() &&
                  top_k.numel() == rows && top_k.is_contiguous(),
              "sm70_sample_topk_candidates_out: per-row tensor mismatch");
  const float* top_p_ptr = nullptr;
  if (top_p.has_value()) {
    TORCH_CHECK(top_p->is_cuda() && top_p->scalar_type() == torch::kFloat32 &&
                    top_p->numel() == rows && top_p->is_contiguous(),
                "sm70_sample_topk_candidates_out: top_p mismatch");
    top_p_ptr = top_p->data_ptr<float>();
  }
  TORCH_CHECK(vocab_size >= kLocalK && vocab_size <= (int64_t(1) << 24),
              "sm70_sample_topk_candidates_out: invalid vocabulary size");
  // A request generator draws its own row (q[i].exponential_(generator));
  // otherwise one default-generator draw covers the [rows, vocab] tensor.
  TORCH_CHECK(!generator.has_value() || rows == 1,
              "sm70_sample_topk_candidates_out: generator needs one row");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(wire));
  const int64_t numel = generator.has_value() ? vocab_size : rows * vocab_size;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const int64_t blocks_per_sm =
      properties->maxThreadsPerMultiProcessor / kNativeBlockSize;
  const int64_t grid_blocks =
      std::min<int64_t>((numel + kNativeBlockSize - 1) / kNativeBlockSize,
                        properties->multiProcessorCount * blocks_per_sm);
  const int64_t total_threads = grid_blocks * kNativeBlockSize;
  const uint64_t counter_offset =
      ((numel - 1) / (total_threads * kNativeUnroll) + 1) *
      kMaxGeneratorOffsetsPerCurandCall;

  auto* cuda_generator = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      generator, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args;
  {
    std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
    philox_args = cuda_generator->philox_cuda_state(counter_offset);
  }

  sample_topk_candidates_kernel<<<rows, kMaxCandidates, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
      sampled.data_ptr<int32_t>(), wire.data_ptr<float>(),
      temperature.data_ptr<float>(), top_k.data_ptr<int32_t>(), top_p_ptr,
      static_cast<int>(rows), static_cast<int>(tp_size), vocab_size,
      generator.has_value() ? 0 : vocab_size, total_threads, philox_args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
