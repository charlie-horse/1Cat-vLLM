// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused decode router for the E256/K6 sigmoid MoE router with an FP32 gate
// weight [256, 2048] and FP16 activations, for one or two tokens.
//
// One launch replaces the four kernels of the generic route (FP16->FP32 cast,
// FP32 GEMM, split-K reduce, topk_sigmoid):
//   1. One CTA per expert row. Each of the 128 threads loads four float4 of
//      the row (the whole row is in flight at once) and the matching FP16
//      activations, converts them exactly to FP32 and accumulates with FP32
//      FMAs. Warp shuffles and a fixed-order sum over the four warps give the
//      logit, which is stored to a per-layer workspace.
//   2. The last CTA to arrive (int arrival counter, re-armed by that CTA, no
//      float atomics) scores every logit and selects the top six per token
//      with the arithmetic of topk_sigmoid's SCORING_SIGMOID path in
//      csrc/moe/topk_softmax_kernels.cu: s = 1 / (1 + __expf(-logit)), NaN or
//      Inf scores become 0, the larger score wins and ties go to the lower
//      expert id, and the weights are renormalized by the sum of the selected
//      scores accumulated in selection order.
// Only the FP32 summation order of the logits differs from the generic route,
// so routing can flip only where two scores tie to within FP32 rounding.
#pragma once
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace vllm::sm70_flint_router {

constexpr int kHidden = 2048;
constexpr int kExperts = 256;
constexpr int kTopK = 6;
constexpr int kMaxTokens = 2;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
// float4 loads of the weight row per thread.
constexpr int kVecs = kHidden / (4 * kThreads);
// Scores held per lane during selection.
constexpr int kPerLane = kExperts / 32;
static_assert(kVecs * 4 * kThreads == kHidden, "row must tile the CTA");
static_assert(kPerLane * 32 == kExperts, "experts must tile a warp");
static_assert(kTopK <= 32 && kMaxTokens <= kWarps, "selection layout");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int d = 16; d; d >>= 1) value += __shfl_xor_sync(0xffffffffu, value, d);
  return value;
}

// topk_sigmoid's score, including its NaN/Inf clamp.
__device__ __forceinline__ float sigmoid_score(float logit) {
  float score = 1.0f / (1.0f + __expf(-logit));
  if (isnan(score) || isinf(score)) score = 0.f;
  return score;
}

// One warp selects one token's top-k from its kExperts logits. Lane l holds
// experts j * 32 + l, so a lane's scan in j order visits ascending ids and a
// strict comparison keeps the lowest id among equal scores; the butterfly
// then breaks ties towards the lower id, as topk_sigmoid does.
__device__ __forceinline__ void select_topk(const float* logits,
                                            float* topk_weights,
                                            int* topk_ids) {
  const int lane = threadIdx.x & 31;
  // Issue every L2 load before the first use.
  float score[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) score[j] = __ldcg(logits + j * 32 + lane);
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) score[j] = sigmoid_score(score[j]);

  float selected_sum = 0.f;
  float own_score = 0.f;
  int own_id = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = score[0];
    int expert = lane;
#pragma unroll
    for (int j = 1; j < kPerLane; ++j) {
      if (score[j] > best) {
        best = score[j];
        expert = j * 32 + lane;
      }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffu, best, d);
      const int other_expert = __shfl_xor_sync(0xffffffffu, expert, d);
      if (other > best || (other == best && other_expert < expert)) {
        best = other;
        expert = other_expert;
      }
    }
    // Every lane holds the same winner; sum in selection order.
    selected_sum += best;
    if (lane == k) {
      own_score = best;
      own_id = expert;
    }
    // Scores lie in [0, 1], so any negative value retires the winner.
#pragma unroll
    for (int j = 0; j < kPerLane; ++j)
      if (expert == j * 32 + lane) score[j] = -10000.f;
  }
  if (lane < kTopK) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    topk_weights[lane] = own_score / denom;
    topk_ids[lane] = own_id;
  }
}

// grid = kExperts CTAs of kThreads. x rows are FP16 [M, kHidden] with an
// element stride that is a multiple of 4 and an 8-byte aligned base; w is the
// contiguous FP32 gate [kExperts, kHidden]. logits is a [kMaxTokens,
// kExperts] FP32 workspace and *counter must be 0 before the first launch.
template <int M>
__global__ void __launch_bounds__(kThreads)
    route_kernel(const half* __restrict__ x, int64_t x_stride,
                 const float* __restrict__ w, float* __restrict__ logits,
                 int* __restrict__ counter, float* __restrict__ topk_weights,
                 int* __restrict__ topk_ids) {
  static_assert(M >= 1 && M <= kMaxTokens, "one or two tokens");
  __shared__ float partial[kWarps][M];
  __shared__ bool is_last;
  const int expert = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const float4* row = reinterpret_cast<const float4*>(
      w + static_cast<int64_t>(expert) * kHidden);
  float4 weight[kVecs];
#pragma unroll
  for (int i = 0; i < kVecs; ++i) weight[i] = __ldg(row + i * kThreads + tid);

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] = 0.f;
    const half* xrow = x + m * x_stride;
#pragma unroll
    for (int i = 0; i < kVecs; ++i) {
      const uint2 raw = __ldg(
          reinterpret_cast<const uint2*>(xrow + (i * kThreads + tid) * 4));
      const float2 lo =
          __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
      const float2 hi =
          __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
      acc[m] = fmaf(weight[i].x, lo.x, acc[m]);
      acc[m] = fmaf(weight[i].y, lo.y, acc[m]);
      acc[m] = fmaf(weight[i].z, hi.x, acc[m]);
      acc[m] = fmaf(weight[i].w, hi.y, acc[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float sum = warp_sum(acc[m]);
    if (lane == 0) partial[warp][m] = sum;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float logit = partial[0][m];
#pragma unroll
      for (int k = 1; k < kWarps; ++k) logit += partial[k][m];
      logits[m * kExperts + expert] = logit;
    }
    // Publish the logits before arriving.
    __threadfence();
    is_last = atomicAdd(counter, 1) == static_cast<int>(gridDim.x) - 1;
  }
  __syncthreads();
  if (!is_last) return;

  // Every other CTA has published its logits. Re-arm the counter for the
  // next launch (which starts only after this kernel completes).
  __threadfence();
  if (tid == 0) *counter = 0;
  if (warp < M)
    select_topk(logits + warp * kExperts, topk_weights + warp * kTopK,
                topk_ids + warp * kTopK);
}

}  // namespace vllm::sm70_flint_router
