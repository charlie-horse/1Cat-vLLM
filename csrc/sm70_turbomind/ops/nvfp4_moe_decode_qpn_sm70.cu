// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
//
// Two-kernel NVFP4 routed-expert decode for M <= 2 tokens on SM70.
//
// The quadpair-N m8n8k4 layout is derived from dnv2003/v100-skinny (MIT), as
// in mxfp4_qpn_m1_sm70.cu. See LICENSE.v100-skinny in this directory.
//
// Both kernels read the TurboMind-packed expert banks in place
// (w13_tm_weight [E, K, N / 8] int32 and w13_tm_scales [E, K / 16, N] FP16,
// likewise for W2) and take routing from device expert IDs, so there is no
// permute, no host synchronization and no data-dependent host branch inside
// a captured graph:
//
//   1. W13 + SwiGLU: one CTA per (route, 32 gate + 32 up columns, K slice).
//      FP32 MMA accumulation. Each CTA joins its warps' K partitions in a
//      fixed order; the last CTA of a (route, tile) to arrive (an integer
//      arrival counter) sums the FP32 K-slice partials in slice order, rounds
//      gate and up to FP16 and applies the eager FP16 SiLU-and-multiply
//      arithmetic. It then re-arms the counter, so the kernel replays from a
//      CUDA graph without a reset launch.
//   2. W2 + weighted reduce: one CTA per (token, 32 output columns); one warp
//      group per routed slot, FP32 accumulation, each routed output rounded
//      to FP16 (the unfused W2 boundary), then the top-k weighted sum in slot
//      order in FP32.
//
// The single-launch op (fused_decode_kernel) runs the same CTAs in one grid:
// W2 CTAs load their weights while W13 runs and wait on a device-side flag
// for their token's activations, which removes the launch boundary and the
// W2 weight latency from the critical path. It is bitwise equal to the two
// kernels.
//
// Every reduction has a fixed order and no floating-point atomics are used,
// so results are deterministic and independent of CTA scheduling. Each warp
// issues all of its weight, scale and activation loads before its first MMA.
//
// Scale overflow guard: the decoder forms prepared_scale * 2^14 in FP16. A
// prepared scale of 4 or more would overflow, so the host passes a power-of-
// two shift s per projection (chosen at load time from the layer's largest
// scale). The kernel then uses prepared_scale * 2^(14 - s) and multiplies the
// FP32 accumulator by 2^s. Both factors are exact, so s = 0 is the unshifted
// arithmetic.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>

namespace {
namespace decode_qpn {

__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                               half2 out[4]) {
  constexpr unsigned kSign = 0x80008000u;
  constexpr unsigned kExponentMantissa = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & kSign) | ((packed << 9) & kExponentMantissa);
  values[1] = ((packed << 8) & kSign) | ((packed << 5) & kExponentMantissa);
  values[2] = ((packed << 4) & kSign) | ((packed << 1) & kExponentMantissa);
  values[3] = (packed & kSign) | ((packed >> 3) & kExponentMantissa);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    out[index] =
        __hmul2(*reinterpret_cast<const half2*>(&values[index]), scale);
  }
}

__device__ __forceinline__ void mma_m8n8k4(float c[8], unsigned a0, unsigned a1,
                                           unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
        "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int packed_column(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

// One warp's packed weights and prepared scales for kGroups K16 groups of a
// 32-column tile, loaded before any of them is consumed.
template <int kGroups>
struct TileFragment {
  unsigned packed[kGroups][2];
  half scale[kGroups];
};

// expert_weights / expert_scales point at one expert's [N/32][K/8][32] words
// and [K/16][N] scales.
template <int kK, int kN, int kGroups>
__device__ __forceinline__ void load_tile(TileFragment<kGroups>& fragment,
                                          const uint32_t* __restrict__ weights,
                                          const half* __restrict__ scales,
                                          int tile, int group_begin, int lane) {
  const int col = packed_column(lane);
#pragma unroll
  for (int index = 0; index < kGroups; ++index) {
    const int group = group_begin + index;
    const size_t base =
        (static_cast<size_t>(tile) * (kK / 8) + group * 2) * 32 + col;
    fragment.packed[index][0] = __ldcs(weights + base);
    fragment.packed[index][1] = __ldcs(weights + base + 32);
    fragment.scale[index] = __ldg(
        scales + (static_cast<size_t>(group) * (kN / 32) + tile) * 32 + col);
  }
}

// The A operand of m8n8k4 carries the single input row on the lanes of row
// 0 (one lane per quadpair); the other rows are zero.
__device__ __forceinline__ bool is_input_lane(int lane) {
  return ((lane & 3) + ((lane & 16) ? 4 : 0)) == 0;
}

// This lane's FP16 input values for kGroups K16 groups, loaded up front.
template <int kGroups>
struct InputFragment {
  uint4 values[kGroups][2];
};

template <int kGroups>
__device__ __forceinline__ void load_input(InputFragment<kGroups>& fragment,
                                           const half* __restrict__ input,
                                           int group_begin, int lane) {
  const bool input_lane = is_input_lane(lane);
#pragma unroll
  for (int index = 0; index < kGroups; ++index) {
    fragment.values[index][0] = make_uint4(0, 0, 0, 0);
    fragment.values[index][1] = make_uint4(0, 0, 0, 0);
    if (input_lane) {
      const half* row = input + (group_begin + index) * 16;
      fragment.values[index][0] = *reinterpret_cast<const uint4*>(row);
      fragment.values[index][1] = *reinterpret_cast<const uint4*>(row + 8);
    }
  }
}

// One K16 group: decode the packed codes with the prepared scale and issue
// the four m8n8k4 MMAs into one FP32 accumulator chain.
__device__ __forceinline__ void mma_group(
    float chain[8], const unsigned packed[2], half group_scale,
    half scale_multiplier, const uint4& input_lo, const uint4& input_hi) {
  const half scalar = __hmul(group_scale, scale_multiplier);
  const half2 scale = __halves2half2(scalar, scalar);
  half2 decoded[8];
  dequant_e2m1x8(packed[0], scale, decoded);
  dequant_e2m1x8(packed[1], scale, decoded + 4);
  const unsigned* b = reinterpret_cast<const unsigned*>(decoded);
  const unsigned* a0 = reinterpret_cast<const unsigned*>(&input_lo);
  const unsigned* a1 = reinterpret_cast<const unsigned*>(&input_hi);
  mma_m8n8k4(chain, a0[0], a0[1], b[0], b[1]);
  mma_m8n8k4(chain, a0[2], a0[3], b[2], b[3]);
  mma_m8n8k4(chain, a1[0], a1[1], b[4], b[5]);
  mma_m8n8k4(chain, a1[2], a1[3], b[6], b[7]);
}

// Accumulate the fragments. Even and odd groups use independent FP32
// accumulator chains so consecutive MMAs do not wait on each other; the two
// are joined once, odd into even, in a fixed order.
template <int kGroups>
__device__ __forceinline__ void mma_tile(float accum[8],
                                         const TileFragment<kGroups>& fragment,
                                         const InputFragment<kGroups>& input,
                                         half scale_multiplier) {
  float chains[2][8] = {};
#pragma unroll
  for (int index = 0; index < kGroups; ++index) {
    mma_group(chains[index & 1], fragment.packed[index], fragment.scale[index],
              scale_multiplier, input.values[index][0], input.values[index][1]);
  }
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    accum[index] = chains[0][index] + chains[1][index];
  }
}

// mma_tile with the input row read from shared memory one group at a time,
// so the activation fragment is never held in registers as a whole. The
// arithmetic (operands, MMA order and accumulator chains) is mma_tile's.
template <int kGroups>
__device__ __forceinline__ void mma_tile_shared(
    float accum[8], const TileFragment<kGroups>& fragment,
    const half* __restrict__ row, int group_begin, int lane,
    half scale_multiplier) {
  const bool input_lane = is_input_lane(lane);
  float chains[2][8] = {};
#pragma unroll
  for (int index = 0; index < kGroups; ++index) {
    uint4 input_lo = make_uint4(0, 0, 0, 0);
    uint4 input_hi = make_uint4(0, 0, 0, 0);
    if (input_lane) {
      const half* group_row = row + (group_begin + index) * 16;
      input_lo = *reinterpret_cast<const uint4*>(group_row);
      input_hi = *reinterpret_cast<const uint4*>(group_row + 8);
    }
    mma_group(chains[index & 1], fragment.packed[index], fragment.scale[index],
              scale_multiplier, input_lo, input_hi);
  }
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    accum[index] = chains[0][index] + chains[1][index];
  }
}

// Scatter row 0 of an m8n8k4 accumulator into a 32-column FP32 row.
__device__ __forceinline__ void store_row0(float* row, const float accum[8],
                                           int lane) {
  if ((lane & 17) == 0) {
    const int quadpair = (lane >> 2) & 3;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
      for (int offset = 0; offset < 2; ++offset) {
        const int col = offset | (((lane >> 1) & 1) << 1) | (pair << 2);
        row[quadpair * 8 + col] = accum[pair * 4 + offset];
      }
    }
  }
}

__device__ __forceinline__ half scale_multiplier(int scale_shift) {
  return __float2half_rn(ldexpf(1.0f, 14 - scale_shift));
}

// FP16 gate/up boundary, then the eager SiLU-and-multiply arithmetic: FP32
// SiLU rounded to FP16, times up in FP16.
__device__ __forceinline__ half swiglu_epilogue(float gate_sum, float up_sum,
                                                int scale_shift) {
  const float epilogue = ldexpf(1.0f, scale_shift);
  const half up = __float2half(up_sum * epilogue);
  const float gate = __half2float(__float2half(gate_sum * epilogue));
  const half silu = __float2half(gate / (1.0f + expf(-gate)));
  return __hmul(silu, up);
}

// Join each routed slot's K warps in order, round the routed output to FP16
// (the unfused W2 boundary), then the top-k weighted sum in slot order in
// FP32.
template <int kTopK, int kSplitWarps>
__device__ __forceinline__ half w2_weighted_sum(
    const float (&warp_rows)[kTopK][kSplitWarps][32],
    const float* __restrict__ token_weights, int lane, int scale_shift) {
  const float epilogue = ldexpf(1.0f, scale_shift);
  float weighted = 0.0f;
#pragma unroll
  for (int routed = 0; routed < kTopK; ++routed) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < kSplitWarps; ++k_warp) {
      value += warp_rows[routed][k_warp][lane];
    }
    const float routed_output = __half2float(__float2half(value * epilogue));
    weighted = fmaf(routed_output, __ldg(token_weights + routed), weighted);
  }
  return __float2half(weighted);
}

// Barrier over the threads of one W13 task: the whole CTA in the two-kernel
// path, one task's warps (named barrier 1 + group, group < 3) in the single
// launch. Immediate barrier IDs keep ptxas from reserving all 16.
template <bool kGrouped, int kThreads>
__device__ __forceinline__ void task_sync(int group) {
  if constexpr (kGrouped) {
    if (group == 0) {
      asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
    } else if (group == 1) {
      asm volatile("bar.sync 2, %0;" ::"n"(kThreads) : "memory");
    } else {
      asm volatile("bar.sync 3, %0;" ::"n"(kThreads) : "memory");
    }
  } else {
    __syncthreads();
  }
}

__device__ __forceinline__ int load_acquire_gpu(const int* address) {
  int value;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

template <int kSplitWarps>
struct W13Shared {
  float warp_rows[2][kSplitWarps][32];
  float joined[2][32];
  bool is_last;
};

// One W13 task: (route, 32 gate + 32 up columns, K slice) on 2 * kSplitWarps
// warps; warp is the warp index within the task.
// output: [routes, I] FP16 SwiGLU activations.
// partials: [routes, I/32, kSplitCta, 2 (gate, up), 32] FP32 and counters
// [routes, I/32] int32 (zero between launches); unused when kSplitCta == 1.
// ready (single-launch path only, else null): per-token count of finished
// (route, tile) activation columns, released after the activation store.
template <int kHidden, int kIntermediate, int kTopK, int kExperts,
          int kSplitWarps, int kSplitCta, bool kGrouped>
__device__ __forceinline__ void w13_swiglu_task(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    half* __restrict__ output, float* __restrict__ partials,
    int* __restrict__ counters, int* __restrict__ ready, int scale_shift,
    int route, int tile, int cta_split, int warp, int lane, int group,
    W13Shared<kSplitWarps>& shared) {
  constexpr int kThreads = 64 * kSplitWarps;
  constexpr int kN = 2 * kIntermediate;
  constexpr int kTiles = kIntermediate / 32;
  constexpr int kGroupsPerWarp = kHidden / 16 / (kSplitWarps * kSplitCta);
  static_assert(kIntermediate % 32 == 0, "intermediate must tile by 32");
  static_assert(kHidden / 16 % (kSplitWarps * kSplitCta) == 0,
                "split-K must divide K / 16");
  auto& warp_rows = shared.warp_rows;
  auto& joined = shared.joined;
  bool& is_last = shared.is_last;

  const int projection = warp / kSplitWarps;
  const int split = warp % kSplitWarps;
  const int expert = __ldg(expert_ids + route);
  const bool valid = expert >= 0 && expert < kExperts;

  const int group_begin = (cta_split * kSplitWarps + split) * kGroupsPerWarp;
  // The input row does not depend on routing: load it alongside the ID.
  InputFragment<kGroupsPerWarp> activation;
  load_input(activation, input + static_cast<size_t>(route / kTopK) * kHidden,
             group_begin, lane);
  float accum[8] = {};
  if (valid) {
    TileFragment<kGroupsPerWarp> fragment;
    load_tile<kHidden, kN>(
        fragment, weights + static_cast<size_t>(expert) * (kHidden * kN / 8),
        scales + static_cast<size_t>(expert) * (kHidden / 16 * kN),
        tile + projection * kTiles, group_begin, lane);
    mma_tile(accum, fragment, activation, scale_multiplier(scale_shift));
  }
  store_row0(warp_rows[projection][split], accum, lane);
  task_sync<kGrouped, kThreads>(group);

  if (warp < 2) {
    float value = 0.0f;
#pragma unroll
    for (int k_warp = 0; k_warp < kSplitWarps; ++k_warp) {
      value += warp_rows[warp][k_warp][lane];
    }
    joined[warp][lane] = value;
  }

  if constexpr (kSplitCta > 1) {
    // The last task of this (route, tile) to arrive sums every K slice in
    // slice order; arrival order never changes the arithmetic.
    const int slot = route * kTiles + tile;
    float* slot_partials =
        partials + static_cast<size_t>(slot) * kSplitCta * 2 * 32;
    if (warp < 2) {
      slot_partials[(cta_split * 2 + warp) * 32 + lane] = joined[warp][lane];
      __threadfence();
    }
    task_sync<kGrouped, kThreads>(group);
    if (warp == 0 && lane == 0) {
      is_last = atomicAdd(counters + slot, 1) == kSplitCta - 1;
    }
    task_sync<kGrouped, kThreads>(group);
    if (!is_last) {
      return;
    }
    if (warp < 2) {
      __threadfence();
      float value = 0.0f;
#pragma unroll
      for (int part = 0; part < kSplitCta; ++part) {
        value += __ldcg(slot_partials + (part * 2 + warp) * 32 + lane);
      }
      joined[warp][lane] = value;
    }
    if (warp == 0 && lane == 0) {
      // Re-arm for the next launch (graph replay or the next layer).
      counters[slot] = 0;
    }
  }
  task_sync<kGrouped, kThreads>(group);

  if (warp == 0) {
    output[static_cast<size_t>(route) * kIntermediate + tile * 32 + lane] =
        swiglu_epilogue(joined[0][lane], joined[1][lane], scale_shift);
    if (ready != nullptr) {
      // Release these 32 activation columns to the W2 CTAs of this token.
      __threadfence();
      __syncwarp();
      if (lane == 0) {
        atomicAdd(ready + route / kTopK, 1);
      }
    }
  }
}

// grid (I/32 * kSplitCta, routes); block 2 * kSplitWarps warps.
template <int kHidden, int kIntermediate, int kTopK, int kExperts,
          int kSplitWarps, int kSplitCta>
__global__ void __launch_bounds__(64 * kSplitWarps)
    w13_swiglu_kernel(const half* __restrict__ input,
                      const uint32_t* __restrict__ weights,
                      const half* __restrict__ scales,
                      const int32_t* __restrict__ expert_ids,
                      half* __restrict__ output, float* __restrict__ partials,
                      int* __restrict__ counters, int scale_shift) {
  constexpr int kTiles = kIntermediate / 32;
  __shared__ W13Shared<kSplitWarps> shared;
  w13_swiglu_task<kHidden, kIntermediate, kTopK, kExperts, kSplitWarps,
                  kSplitCta, false>(
      input, weights, scales, expert_ids, output, partials, counters, nullptr,
      scale_shift, blockIdx.y, blockIdx.x % kTiles, blockIdx.x / kTiles,
      threadIdx.x >> 5, threadIdx.x & 31, 0, shared);
}

// grid (hidden / 32, tokens); block kTopK * kSplitWarps warps.
template <int kHidden, int kIntermediate, int kTopK, int kExperts,
          int kSplitWarps>
__global__ void __launch_bounds__(32 * kTopK * kSplitWarps)
    w2_reduce_kernel(const half* __restrict__ input,
                     const uint32_t* __restrict__ weights,
                     const half* __restrict__ scales,
                     const int32_t* __restrict__ expert_ids,
                     const float* __restrict__ topk_weights,
                     half* __restrict__ output, int scale_shift) {
  constexpr int kGroupsPerWarp = kIntermediate / 16 / kSplitWarps;
  static_assert(kIntermediate / 16 % kSplitWarps == 0,
                "split-K must divide K / 16");
  __shared__ float warp_rows[kTopK][kSplitWarps][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = warp / kSplitWarps;
  const int split = warp % kSplitWarps;
  const int token = blockIdx.y;
  const int tile = blockIdx.x;
  const int route = token * kTopK + slot;
  const int expert = __ldg(expert_ids + route);
  const int group_begin = split * kGroupsPerWarp;

  InputFragment<kGroupsPerWarp> activation;
  load_input(activation, input + static_cast<size_t>(route) * kIntermediate,
             group_begin, lane);
  float accum[8] = {};
  if (expert >= 0 && expert < kExperts) {
    TileFragment<kGroupsPerWarp> fragment;
    load_tile<kIntermediate, kHidden>(
        fragment,
        weights + static_cast<size_t>(expert) * (kIntermediate * kHidden / 8),
        scales + static_cast<size_t>(expert) * (kIntermediate / 16 * kHidden),
        tile, group_begin, lane);
    mma_tile(accum, fragment, activation, scale_multiplier(scale_shift));
  }
  store_row0(warp_rows[slot][split], accum, lane);
  __syncthreads();

  if (warp == 0) {
    output[static_cast<size_t>(token) * kHidden + tile * 32 + lane] =
        w2_weighted_sum<kTopK, kSplitWarps>(
            warp_rows, topk_weights + token * kTopK, lane, scale_shift);
  }
}

// Single-launch decode: W13 + SwiGLU and W2 + top-k reduce in one grid, so
// the W2 weights stream in while W13 runs and no launch boundary separates
// the stages. Every block has kTopK * kW2SplitWarps warps.
//
//   * Blocks [0, w13_blocks): W13 tasks, each 2 * kW13SplitWarps warps and
//     exactly one CTA of w13_swiglu_kernel (same task index -> (route, tile,
//     K slice) mapping, same arithmetic, named barriers per task).
//   * Blocks [w13_blocks, w13_blocks + w2_blocks): one W2 tile per block,
//     exactly a CTA of w2_reduce_kernel. Each block loads its expert IDs and
//     W2 weight fragments first, then waits until its token's activations are
//     complete (an integer flag the W13 epilogues release), stages the six
//     activation rows in shared memory and finishes as w2_reduce_kernel does.
//
// Forward progress: only W2 blocks wait, and only on W13 tasks. The host
// launches this kernel only when more blocks fit on the device at once than
// there are W2 blocks, so at least one block slot always runs W13 work even
// if every W2 block were resident and waiting; W2 blocks also come last in
// launch order, after every W13 block. The flags re-arm in the kernel: the
// last W2 block to pass its wait resets them, so graph replays need no reset.
//
// counters: [kArrivals] W13 split-K arrivals, then [kMaxTokens] ready flags
// and one exit ticket, all zero between launches.
template <int kHidden, int kIntermediate, int kTopK, int kExperts,
          int kW13SplitWarps, int kW13SplitCta, int kW2SplitWarps,
          int kMaxTokens, int kMinBlocks>
__global__ void __launch_bounds__(32 * kTopK * kW2SplitWarps, kMinBlocks)
    fused_decode_kernel(const half* __restrict__ input,
                        const uint32_t* __restrict__ w13_weights,
                        const half* __restrict__ w13_scales,
                        const uint32_t* __restrict__ w2_weights,
                        const half* __restrict__ w2_scales,
                        const int32_t* __restrict__ expert_ids,
                        const float* __restrict__ topk_weights,
                        half* __restrict__ activations,
                        half* __restrict__ output, float* __restrict__ partials,
                        int* __restrict__ counters, int w13_blocks,
                        int w2_blocks, int w13_scale_shift,
                        int w2_scale_shift) {
  constexpr int kThreads = 32 * kTopK * kW2SplitWarps;
  constexpr int kTaskWarps = 2 * kW13SplitWarps;
  constexpr int kTasksPerBlock = kThreads / (32 * kTaskWarps);
  constexpr int kTiles = kIntermediate / 32;
  constexpr int kArrivals = kMaxTokens * kTopK * kTiles;
  constexpr int kW2Tiles = kHidden / 32;
  constexpr int kW2Groups = kIntermediate / 16 / kW2SplitWarps;
  constexpr int kActivationVectors = kTopK * kIntermediate / 8;
  static_assert(kThreads % (32 * kTaskWarps) == 0,
                "a block must hold whole W13 tasks");
  static_assert(kTasksPerBlock <= 3, "task_sync has three named barriers");
  static_assert(kTopK * kTiles * kW13SplitCta % kTasksPerBlock == 0,
                "a token's W13 tasks must fill whole blocks");
  static_assert(kIntermediate / 16 % kW2SplitWarps == 0,
                "split-K must divide K / 16");
  static_assert(kActivationVectors <= kThreads, "one vector per thread");
  __shared__ W13Shared<kW13SplitWarps> w13_shared[kTasksPerBlock];
  __shared__ float w2_rows[kTopK][kW2SplitWarps][32];
  __shared__ uint4 staged[kActivationVectors];

  int* ready = counters + kArrivals;
  int* exit_ticket = ready + kMaxTokens;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  if (static_cast<int>(blockIdx.x) < w13_blocks) {
    const int group = warp / kTaskWarps;
    const int task = blockIdx.x * kTasksPerBlock + group;
    const int route = task / (kTiles * kW13SplitCta);
    const int column_block = task % (kTiles * kW13SplitCta);
    w13_swiglu_task<kHidden, kIntermediate, kTopK, kExperts, kW13SplitWarps,
                    kW13SplitCta, true>(
        input, w13_weights, w13_scales, expert_ids, activations, partials,
        counters, ready, w13_scale_shift, route, column_block % kTiles,
        column_block / kTiles, warp % kTaskWarps, lane, group,
        w13_shared[group]);
    return;
  }

  const int w2_block = blockIdx.x - w13_blocks;
  const int token = w2_block / kW2Tiles;
  const int tile = w2_block % kW2Tiles;
  const int slot = warp / kW2SplitWarps;
  const int split = warp % kW2SplitWarps;
  const int route = token * kTopK + slot;
  const int expert = __ldg(expert_ids + route);
  const bool valid = expert >= 0 && expert < kExperts;
  const int group_begin = split * kW2Groups;

  // Issue this block's W2 weight loads before waiting on W13.
  TileFragment<kW2Groups> fragment;
  if (valid) {
    load_tile<kIntermediate, kHidden>(
        fragment,
        w2_weights +
            static_cast<size_t>(expert) * (kIntermediate * kHidden / 8),
        w2_scales +
            static_cast<size_t>(expert) * (kIntermediate / 16 * kHidden),
        tile, group_begin, lane);
  }

  int ticket = 0;
  if (threadIdx.x == 0) {
    const int complete = kTopK * kTiles;
    while (load_acquire_gpu(ready + token) < complete) {
      __nanosleep(32);
    }
    ticket = atomicAdd(exit_ticket, 1);
  }
  __syncthreads();

  // The activations were written by other blocks of this launch: read them
  // at L2 (never a stale L1 line) into shared memory.
  if (threadIdx.x < kActivationVectors) {
    staged[threadIdx.x] = __ldcg(
        reinterpret_cast<const uint4*>(
            activations + static_cast<size_t>(token) * kTopK * kIntermediate) +
        threadIdx.x);
  }
  __syncthreads();

  float accum[8] = {};
  if (valid) {
    mma_tile_shared(
        accum, fragment,
        reinterpret_cast<const half*>(staged) + slot * kIntermediate,
        group_begin, lane, scale_multiplier(w2_scale_shift));
  }
  store_row0(w2_rows[slot][split], accum, lane);
  __syncthreads();

  if (warp == 0) {
    output[static_cast<size_t>(token) * kHidden + tile * 32 + lane] =
        w2_weighted_sum<kTopK, kW2SplitWarps>(
            w2_rows, topk_weights + token * kTopK, lane, w2_scale_shift);
  }
  if (threadIdx.x == 0 && ticket == w2_blocks - 1) {
    // Every W2 block has passed its wait, so every ready flag is final:
    // re-arm them for the next launch.
#pragma unroll
    for (int index = 0; index < kMaxTokens; ++index) {
      ready[index] = 0;
    }
    *exit_ticket = 0;
  }
}

// Flint TP2: hidden 2048, local intermediate 256, top-6 of 256 experts.
constexpr int kFlintHidden = 2048;
constexpr int kFlintIntermediate = 256;
constexpr int kFlintTopK = 6;
constexpr int kFlintExperts = 256;
constexpr int kFlintTiles = kFlintIntermediate / 32;
constexpr int kFlintMaxTokens = 2;
// Tuned on V100-PCIE-32GB (CUDA graph, 8 rotating routings): W13 runs 16 K
// slices of 128 per (route, tile) with two warps per projection (four K16
// groups per warp); W2 runs two warps per routed slot.
constexpr int kFlintW13SplitWarps = 2;
constexpr int kFlintW13SplitCta = 16;
constexpr int kFlintW2SplitWarps = 2;
// Split-K workspace in elements. The caller allocates these sizes
// (nvfp4_sm70_moe.py: _DECODE_QPN_PARTIALS and _DECODE_QPN_COUNTERS): the
// W13 arrival counters, then the single-launch flags padded to 128 bytes.
constexpr int64_t kFlintCounters = kFlintMaxTokens * kFlintTopK * kFlintTiles;
constexpr int64_t kFlintSyncCounters = 32;
constexpr int64_t kFlintPartials = kFlintCounters * kFlintW13SplitCta * 2 * 32;
static_assert(kFlintCounters == 96 && kFlintPartials == 98304,
              "update the Python workspace sizes with the split-K tuning");
static_assert(kFlintSyncCounters >= kFlintMaxTokens + 1,
              "ready flags and the exit ticket must fit");
// Single launch: 384-thread blocks (three W13 tasks or one W2 tile each),
// at least two resident per SM so the W2 blocks of two tokens (128) never
// fill the device (80 SMs on a V100).
constexpr int kFlintFusedThreads = 32 * kFlintTopK * kFlintW2SplitWarps;
constexpr int kFlintFusedMinBlocks = 2;
constexpr int kFlintW13TasksPerBlock =
    kFlintFusedThreads / (64 * kFlintW13SplitWarps);
constexpr int kFlintW2Tiles = kFlintHidden / 32;

#define FLINT_FUSED_DECODE_KERNEL                                            \
  fused_decode_kernel<kFlintHidden, kFlintIntermediate, kFlintTopK,          \
                      kFlintExperts, kFlintW13SplitWarps, kFlintW13SplitCta, \
                      kFlintW2SplitWarps, kFlintMaxTokens,                   \
                      kFlintFusedMinBlocks>

void check_bank(const char* name, const torch::Tensor& input,
                const torch::Tensor& weights, const torch::Tensor& scales,
                const torch::Tensor& expert_ids, const torch::Tensor& out,
                int64_t scale_shift) {
  TORCH_CHECK(input.is_cuda() && weights.is_cuda() && scales.is_cuda() &&
                  expert_ids.is_cuda() && out.is_cuda(),
              name, ": tensors must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16 &&
                  weights.scalar_type() == torch::kInt32 &&
                  scales.scalar_type() == torch::kFloat16 &&
                  expert_ids.scalar_type() == torch::kInt32,
              name, ": dtype mismatch");
  TORCH_CHECK(input.is_contiguous() && weights.is_contiguous() &&
                  scales.is_contiguous() && expert_ids.is_contiguous() &&
                  out.is_contiguous(),
              name, ": tensors must be contiguous");
  TORCH_CHECK(input.get_device() == weights.get_device() &&
                  input.get_device() == scales.get_device() &&
                  input.get_device() == expert_ids.get_device() &&
                  input.get_device() == out.get_device(),
              name, ": device mismatch");
  TORCH_CHECK(scale_shift >= 0 && scale_shift <= 14, name,
              ": scale_shift must be in [0, 14]");
  const auto* properties = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0, name,
              ": requires SM70");
}

// W13 bank and split-K workspace; returns the token count.
int64_t check_w13(const char* name, const torch::Tensor& out,
                  const torch::Tensor& input, const torch::Tensor& weights,
                  const torch::Tensor& scales, const torch::Tensor& expert_ids,
                  const torch::Tensor& partials, const torch::Tensor& counters,
                  int64_t scale_shift, int64_t min_counters) {
  check_bank(name, input, weights, scales, expert_ids, out, scale_shift);
  const int64_t tokens = input.size(0);
  const int64_t routes = tokens * kFlintTopK;
  TORCH_CHECK(
      input.dim() == 2 && (tokens == 1 || tokens == 2) &&
          input.size(1) == kFlintHidden &&
          out.sizes() == torch::IntArrayRef({routes, kFlintIntermediate}) &&
          weights.sizes() == torch::IntArrayRef({kFlintExperts, kFlintHidden,
                                                 2 * kFlintIntermediate / 8}) &&
          scales.sizes() ==
              torch::IntArrayRef(
                  {kFlintExperts, kFlintHidden / 16, 2 * kFlintIntermediate}) &&
          expert_ids.numel() == routes,
      name, ": expected Flint TP2 W13 tensors (M1/M2, H2048, I256, E256, K6)");
  TORCH_CHECK(partials.is_cuda() && counters.is_cuda() &&
                  partials.scalar_type() == torch::kFloat32 &&
                  counters.scalar_type() == torch::kInt32 &&
                  partials.is_contiguous() && counters.is_contiguous() &&
                  partials.get_device() == input.get_device() &&
                  counters.get_device() == input.get_device() &&
                  partials.numel() == kFlintPartials &&
                  counters.numel() >= min_counters,
              name, ": split-K workspace must be contiguous CUDA FP32 [",
              kFlintPartials, "] and zero-initialized int32 [>= ", min_counters,
              "]");
  return tokens;
}

void check_w2(const char* name, const torch::Tensor& out,
              const torch::Tensor& input, const torch::Tensor& weights,
              const torch::Tensor& scales, const torch::Tensor& expert_ids,
              const torch::Tensor& topk_weights, int64_t scale_shift) {
  check_bank(name, input, weights, scales, expert_ids, out, scale_shift);
  TORCH_CHECK(topk_weights.is_cuda() &&
                  topk_weights.scalar_type() == torch::kFloat32 &&
                  topk_weights.is_contiguous() &&
                  topk_weights.get_device() == input.get_device(),
              name, ": topk_weights must be contiguous CUDA FP32");
  const int64_t tokens = out.size(0);
  TORCH_CHECK(
      out.dim() == 2 && (tokens == 1 || tokens == 2) &&
          out.size(1) == kFlintHidden &&
          input.sizes() ==
              torch::IntArrayRef({tokens * kFlintTopK, kFlintIntermediate}) &&
          weights.sizes() ==
              torch::IntArrayRef(
                  {kFlintExperts, kFlintIntermediate, kFlintHidden / 8}) &&
          scales.sizes() ==
              torch::IntArrayRef(
                  {kFlintExperts, kFlintIntermediate / 16, kFlintHidden}) &&
          expert_ids.numel() == tokens * kFlintTopK &&
          topk_weights.sizes() == torch::IntArrayRef({tokens, kFlintTopK}),
      name, ": expected Flint TP2 W2 tensors (M1/M2, H2048, I256, E256, K6)");
}

void launch_w13(const torch::Tensor& out, const torch::Tensor& input,
                const torch::Tensor& weights, const torch::Tensor& scales,
                const torch::Tensor& expert_ids, const torch::Tensor& partials,
                const torch::Tensor& counters, int64_t tokens,
                int64_t scale_shift) {
  w13_swiglu_kernel<kFlintHidden, kFlintIntermediate, kFlintTopK, kFlintExperts,
                    kFlintW13SplitWarps, kFlintW13SplitCta>
      <<<dim3(kFlintTiles * kFlintW13SplitCta,
              static_cast<unsigned>(tokens * kFlintTopK)),
         64 * kFlintW13SplitWarps, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          expert_ids.data_ptr<int32_t>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          partials.data_ptr<float>(), counters.data_ptr<int32_t>(),
          static_cast<int>(scale_shift));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_w2(const torch::Tensor& out, const torch::Tensor& input,
               const torch::Tensor& weights, const torch::Tensor& scales,
               const torch::Tensor& expert_ids,
               const torch::Tensor& topk_weights, int64_t tokens,
               int64_t scale_shift) {
  w2_reduce_kernel<kFlintHidden, kFlintIntermediate, kFlintTopK, kFlintExperts,
                   kFlintW2SplitWarps>
      <<<dim3(kFlintHidden / 32, static_cast<unsigned>(tokens)),
         32 * kFlintTopK * kFlintW2SplitWarps, 0,
         at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
          reinterpret_cast<const uint32_t*>(weights.data_ptr<int32_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          expert_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          static_cast<int>(scale_shift));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Blocks of the single-launch kernel that fit on the current device at once
// (cached per device; a pure occupancy query, legal during graph capture).
int fused_resident_blocks(int device) {
  static std::atomic<int> cache[64];
  if (device >= 0 && device < 64) {
    const int cached = cache[device].load(std::memory_order_relaxed);
    if (cached > 0) {
      return cached;
    }
  }
  int per_sm = 0;
  AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, FLINT_FUSED_DECODE_KERNEL, kFlintFusedThreads, 0));
  const int resident =
      per_sm * at::cuda::getDeviceProperties(device)->multiProcessorCount;
  if (device >= 0 && device < 64 && resident > 0) {
    cache[device].store(resident, std::memory_order_relaxed);
  }
  return resident;
}

}  // namespace decode_qpn
}  // namespace

void nvfp4_moe_decode_w13_swiglu_sm70_out(
    torch::Tensor out, torch::Tensor input, torch::Tensor weights,
    torch::Tensor scales, torch::Tensor expert_ids, torch::Tensor partials,
    torch::Tensor counters, int64_t scale_shift) {
  using namespace decode_qpn;
  constexpr const char* kName = "nvfp4_moe_decode_w13_swiglu_sm70_out";
  const int64_t tokens =
      check_w13(kName, out, input, weights, scales, expert_ids, partials,
                counters, scale_shift, kFlintCounters);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  launch_w13(out, input, weights, scales, expert_ids, partials, counters,
             tokens, scale_shift);
}

void nvfp4_moe_decode_w2_reduce_sm70_out(torch::Tensor out, torch::Tensor input,
                                         torch::Tensor weights,
                                         torch::Tensor scales,
                                         torch::Tensor expert_ids,
                                         torch::Tensor topk_weights,
                                         int64_t scale_shift) {
  using namespace decode_qpn;
  constexpr const char* kName = "nvfp4_moe_decode_w2_reduce_sm70_out";
  check_w2(kName, out, input, weights, scales, expert_ids, topk_weights,
           scale_shift);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  launch_w2(out, input, weights, scales, expert_ids, topk_weights, out.size(0),
            scale_shift);
}

// Single-launch W13 + SwiGLU + W2 + top-k reduce, bitwise equal to the two
// ops above. activations is the [routes, I] FP16 intermediate both paths
// write. Falls back to the two launches when the device cannot hold more
// blocks at once than there are W2 blocks.
void nvfp4_moe_decode_fused_sm70_out(
    torch::Tensor out, torch::Tensor activations, torch::Tensor input,
    torch::Tensor w13_weights, torch::Tensor w13_scales,
    torch::Tensor w2_weights, torch::Tensor w2_scales, torch::Tensor expert_ids,
    torch::Tensor topk_weights, torch::Tensor partials, torch::Tensor counters,
    int64_t w13_scale_shift, int64_t w2_scale_shift) {
  using namespace decode_qpn;
  constexpr const char* kName = "nvfp4_moe_decode_fused_sm70_out";
  const int64_t tokens = check_w13(
      kName, activations, input, w13_weights, w13_scales, expert_ids, partials,
      counters, w13_scale_shift, kFlintCounters + kFlintSyncCounters);
  check_w2(kName, out, activations, w2_weights, w2_scales, expert_ids,
           topk_weights, w2_scale_shift);
  TORCH_CHECK(out.size(0) == tokens, kName, ": token count mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const int w13_blocks =
      static_cast<int>(tokens * kFlintTopK * kFlintTiles * kFlintW13SplitCta /
                       kFlintW13TasksPerBlock);
  const int w2_blocks = static_cast<int>(tokens * kFlintW2Tiles);
  if (fused_resident_blocks(input.get_device()) <= w2_blocks) {
    launch_w13(activations, input, w13_weights, w13_scales, expert_ids,
               partials, counters, tokens, w13_scale_shift);
    launch_w2(out, activations, w2_weights, w2_scales, expert_ids, topk_weights,
              tokens, w2_scale_shift);
    return;
  }
  FLINT_FUSED_DECODE_KERNEL<<<w13_blocks + w2_blocks, kFlintFusedThreads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w13_weights.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(w13_scales.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w2_weights.data_ptr<int32_t>()),
      reinterpret_cast<const half*>(w2_scales.data_ptr<at::Half>()),
      expert_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
      reinterpret_cast<half*>(activations.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      partials.data_ptr<float>(), counters.data_ptr<int32_t>(), w13_blocks,
      w2_blocks, static_cast<int>(w13_scale_shift),
      static_cast<int>(w2_scale_shift));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
