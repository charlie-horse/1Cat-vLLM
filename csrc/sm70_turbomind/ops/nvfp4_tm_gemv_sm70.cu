// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Single-token (M <= 2) NVFP4 GEMV over the resident TurboMind SM70 weights.
//
// It reads the TurboMind HMMA884 B/Pack1 codes ([N/32, K/8, 32] u32 words,
// nibble order [0,2,4,6,1,3,5,7], see nvfp4_qpn2_layout.cuh) and the
// TurboMind FP16 group scales ([K/16, N] row-major, global scale already
// folded in) directly: no second weight layout, no workspace, no allocation.
//
// Each lane loads one uint4: four adjacent columns of one K8 row. The lanes
// of an mma.m8n8k4 quad pair supply both the B columns and the A rows, so a
// quad pair can work on G different K8 rows at once: group g owns A rows and
// B columns [g * 8 / G, (g + 1) * 8 / G), feeds its own activations as A and
// only the diagonal blocks of D are kept. M <= 2 leaves enough A rows for
// G = 4. A CTA then covers 32 / G columns over all of K, so small-N layers
// still fill the GPU without a cross-CTA reduction. Partial sums are
// reduced in a fixed order (shared memory, then a shuffle tree), so results
// are deterministic. Rounds can be double buffered in registers: a warp's
// next loads are in flight while it decodes the current ones.
//
// On V100 these single-token GEMVs are bound by per-kernel latency and SM
// work, not DRAM: the same kernel with every weight resident in L2 is barely
// faster. A plain streaming read of the same bytes reaches 260-700 GB/s for
// these 0.7-8 MB layers.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef VLLM_NVFP4_TM_GEMV_STANDALONE
void nvfp4_gemm_sm70_out(torch::Tensor out, torch::Tensor input,
                         torch::Tensor weight, torch::Tensor scales,
                         int64_t group_size, int64_t k_ld, int64_t q_ld,
                         bool gated_silu);
#endif

namespace {

constexpr int kTmGemvMaxRows = 2;

// Per-shape launch table: (K, N, groups, warps, prefetch, pipelined,
// min_blocks), measured on V100-PCIE-32GB with cold L2 under CUDA graph
// replay (benchmarks/kernels/benchmark_sm70_nvfp4_m1.py). See kTmGemvTable.
#define VLLM_TM_GEMV_TABLE(X)                                  \
  X(2048, 2560, 2, 8, 2, true, 1)  /* qkv */                   \
  X(2048, 2048, 2, 8, 2, true, 1)  /* o */                     \
  X(2048, 1024, 2, 8, 2, true, 1)  /* shared expert gate/up */ \
  X(512, 2048, 1, 8, 2, true, 1)   /* shared expert down */    \
  X(2048, 6144, 2, 4, 2, false, 8) /* dense MLP gate/up */     \
  X(3072, 2048, 2, 8, 2, true, 1)  /* dense MLP down */

#define VLLM_SM70_TM_GEMV_MMA(C, A0, A1, B0, B1)                    \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// E2M1 x8 -> four half2 (k0,k1) (k2,k3) (k4,k5) (k6,k7) times `scale`.
// The FP16 bit pattern of each code is the E2M1 value times 2^-14; `scale`
// carries the 2^14 (or 2^shift) factor, so one HMUL2 yields the weight.
//
// Nibble p of `packed` holds K index [0,2,4,6,1,3,5,7][p]; half2 i pairs
// nibbles i and i + 4 (16 bits apart). An FP16 needs the E2M1 sign 3 bits
// further above its exponent/mantissa than the nibble has it, so for every
// other nibble the sign is first moved up 3 bits, into the free nibble
// beside it (one LOP3 select each for even and odd nibbles). Each half2 is
// then one shift and one mask: 13 integer operations per 8 weights instead
// of 16, since LOP3 takes a single immediate.
__device__ __forceinline__ void decode_e2m1x8(unsigned packed, half2 scale,
                                              unsigned out[4]) {
  constexpr unsigned kHalfBits = 0x8e008e00u;  // sign + 3 exponent/mantissa
  constexpr unsigned kMovedSign = 0x40404040u;
  // Even nibbles (bytes' low halves) and odd nibbles shifted onto them.
  const unsigned even = (packed & ~kMovedSign) | ((packed << 3) & kMovedSign);
  const unsigned odd =
      ((packed >> 4) & ~kMovedSign) | ((packed >> 1) & kMovedSign);
  unsigned values[4];
  values[0] = (even << 9) & kHalfBits;  // nibbles 0, 4: k0, k1
  values[1] = (odd << 9) & kHalfBits;   // nibbles 1, 5: k2, k3
  values[2] = (even << 1) & kHalfBits;  // nibbles 2, 6: k4, k5
  values[3] = (odd << 1) & kHalfBits;   // nibbles 3, 7: k6, k7
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const half2 w = __hmul2(*reinterpret_cast<const half2*>(&values[i]), scale);
    out[i] = *reinterpret_cast<const unsigned*>(&w);
  }
}

// One round of a warp: `Prefetch` units of codes, scales and activations.
template <int Prefetch>
struct TmGemvStage {
  uint4 code[Prefetch];
  uint2 scale[Prefetch];
  uint4 a[Prefetch];
};

// mma.m8n8k4 accumulator element `i` of `lane` holds D[row][col] with
// row = (i & 2) + (lane & 16 ? 4 : 0) + (lane & 1) and
// col = (i & 1) | ((lane >> 1) & 1) << 1 | (i >> 2) << 2.
__device__ __forceinline__ int tm_gemv_d_row(int lane, int i) {
  return (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
}
__device__ __forceinline__ int tm_gemv_d_col(int lane, int i) {
  return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

// Largest power of two <= min(32, max_lanes) that divides `partials`: the
// lanes that share one output in the final reduction.
__host__ __device__ constexpr int tm_gemv_lanes_per_output(int max_lanes,
                                                           int partials) {
  int lanes = 1;
  while (lanes * 2 <= 32 && lanes * 2 <= max_lanes &&
         partials % (lanes * 2) == 0) {
    lanes *= 2;
  }
  return lanes;
}

// Whether element `i` can belong to a kept (diagonal, row < M) block for
// some lane; the others are never reduced.
template <int M, int G>
__host__ __device__ constexpr bool tm_gemv_element_used(int i) {
  // Rows within a group: 8 / G. For G <= 2 only rows with (i & 2) == 0 can
  // be below M = 2; with G = 4 every element may be a kept one.
  return G == 4 || (i & 2) == 0;
}

// grid = N / (32 / G) column blocks; block = Warps * 32. A unit is one warp
// load: 4 * G K8 rows of 32 / G columns (512 bytes of codes). Each warp runs
// `rounds` rounds of Prefetch units; Pipelined double-buffers the rounds,
// otherwise a round's loads all issue before its decode. MinBlocks bounds
// registers so that many small CTAs share an SM.
template <int M, int G, int Warps, int Prefetch, bool Pipelined, int MinBlocks>
__global__ __launch_bounds__(
    Warps * 32,
    MinBlocks) void nvfp4_tm_gemv_sm70_kernel(const uint4* __restrict__ codes,
                                              const half* __restrict__ scales,
                                              const half* __restrict__ input,
                                              half* __restrict__ output, int n,
                                              int k, int input_stride,
                                              int rounds, half2 scale_mul,
                                              float out_mul) {
  static_assert(M >= 1 && M <= kTmGemvMaxRows, "TM GEMV supports M <= 2");
  static_assert(G == 1 || G == 2 || G == 4, "TM GEMV groups: 1, 2 or 4");
  static_assert(M <= 8 / G, "Not enough A rows per group");
  constexpr int kGroupSlots = 8 / G;  // B columns and A rows per group
  constexpr int kColumns = 32 / G;    // output columns per CTA
  // Quad pairs and their G groups hold different K8 rows of the same
  // columns; every (warp, quad pair, group) is one partial of each output.
  // Output-major with one pad word: conflict-free stores and reads.
  constexpr int kOutputs = M * kColumns;
  constexpr int kPartials = Warps * 4 * G;
  __shared__ float reduce[kOutputs][kPartials + 1];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quadpair = (lane >> 2) & 3;
  // Slot of this lane among its quad pair's 8 B columns / A rows.
  const int slot = (lane & 3) | ((lane >> 4) << 2);
  const int group = slot / kGroupSlots;
  const int group_slot = slot % kGroupSlots;
  const bool a_lane = group_slot < M;
  const int k8_rows = k >> 3;
  const int tile = blockIdx.x / G;
  const int column_base = tile * 32 + (blockIdx.x % G) * kColumns;

  // Unit u covers K8 rows [4 * G * u, 4 * G * (u + 1)); this lane's row is
  // 4 * G * u + quadpair * G + group.
  const int k8_lane = quadpair * G + group;
  const uint4* lane_codes = codes + static_cast<size_t>(tile) * k8_rows * 8 +
                            (blockIdx.x % G) * kGroupSlots + group_slot +
                            static_cast<size_t>(k8_lane) * 8;
  const half* lane_scales = scales + column_base + group_slot * 4;
  const half* lane_input =
      input + static_cast<size_t>(a_lane ? group_slot : 0) * input_stride +
      k8_lane * 8;

  float accum[4][8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[w][i] = 0.0f;
    }
  }

  TmGemvStage<Prefetch> stages[Pipelined ? 2 : 1];
  auto load_round = [&](TmGemvStage<Prefetch>& stage, int round) {
#pragma unroll
    for (int p = 0; p < Prefetch; ++p) {
      const int unit = (round * Prefetch + p) * Warps + warp;
      const int k8 = unit * 4 * G;
      stage.code[p] = __ldcs(lane_codes + static_cast<size_t>(k8) * 8);
      stage.scale[p] = __ldcs(reinterpret_cast<const uint2*>(
          lane_scales + static_cast<size_t>((k8 + k8_lane) >> 1) * n));
      stage.a[p] =
          a_lane ? __ldg(reinterpret_cast<const uint4*>(lane_input + k8 * 8))
                 : make_uint4(0, 0, 0, 0);
    }
  };
  auto compute_round = [&](const TmGemvStage<Prefetch>& stage) {
#pragma unroll
    for (int p = 0; p < Prefetch; ++p) {
      const half2 s01 = __hmul2(
          *reinterpret_cast<const half2*>(&stage.scale[p].x), scale_mul);
      const half2 s23 = __hmul2(
          *reinterpret_cast<const half2*>(&stage.scale[p].y), scale_mul);
      const unsigned words[4] = {stage.code[p].x, stage.code[p].y,
                                 stage.code[p].z, stage.code[p].w};
      const half2 column_scales[4] = {__low2half2(s01), __high2half2(s01),
                                      __low2half2(s23), __high2half2(s23)};
      const uint4 a = stage.a[p];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        unsigned b[4];
        decode_e2m1x8(words[w], column_scales[w], b);
        VLLM_SM70_TM_GEMV_MMA(accum[w], a.x, a.y, b[0], b[1]);
        VLLM_SM70_TM_GEMV_MMA(accum[w], a.z, a.w, b[2], b[3]);
      }
    }
  };
  if constexpr (Pipelined) {
    load_round(stages[0], 0);
#pragma unroll 1
    for (int round = 0; round < rounds; round += 2) {
      if (round + 1 < rounds) {
        load_round(stages[Pipelined ? 1 : 0], round + 1);
      }
      compute_round(stages[0]);
      if (round + 1 < rounds) {
        if (round + 2 < rounds) {
          load_round(stages[0], round + 2);
        }
        compute_round(stages[Pipelined ? 1 : 0]);
      }
    }
  } else {
#pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
      load_round(stages[0], round);
      compute_round(stages[0]);
    }
  }

  // Each kept accumulator element covers four adjacent output columns,
  // one per word.
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (tm_gemv_element_used<M, G>(i)) {
      const int row = tm_gemv_d_row(lane, i);
      const int col = tm_gemv_d_col(lane, i);
      if (row / kGroupSlots == col / kGroupSlots && row % kGroupSlots < M) {
        const int partial = (warp * 4 + quadpair) * G + row / kGroupSlots;
        const int first =
            (row % kGroupSlots) * kColumns + (col % kGroupSlots) * 4;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          reduce[first + w][partial] = accum[w][i];
        }
      }
    }
  }
  __syncthreads();
  // Several lanes per output, then a shuffle tree; the order is fixed.
  constexpr int kLanesPerOutput =
      tm_gemv_lanes_per_output(Warps * 32 / kOutputs, kPartials);
  constexpr int kPerLane = kPartials / kLanesPerOutput;
  const int output_index = threadIdx.x / kLanesPerOutput;
  const int part = threadIdx.x % kLanesPerOutput;
  float sum = 0.0f;
  if (output_index < kOutputs) {
#pragma unroll
    for (int p = 0; p < kPerLane; ++p) {
      sum += reduce[output_index][p * kLanesPerOutput + part];
    }
  }
#pragma unroll
  for (int offset = kLanesPerOutput / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  }
  if (output_index < kOutputs && part == 0) {
    const int row = output_index / kColumns;
    const int column = output_index % kColumns;
    output[static_cast<size_t>(row) * n + column_base + column] =
        __float2half_rn(sum * out_mul);
  }
}

struct TmGemvConfig {
  int groups;
  int warps;
  int prefetch;
  bool pipelined;
  int min_blocks;
};

template <int M, int G, int Warps, int Prefetch, bool Pipelined, int MinBlocks>
void launch_tm_gemv(const uint4* codes, const half* scales, const half* input,
                    half* output, int n, int k, int input_stride,
                    half2 scale_mul, float out_mul, cudaStream_t stream) {
  const int units = k / (32 * G);
  const int rounds = units / (Warps * Prefetch);
  nvfp4_tm_gemv_sm70_kernel<M, G, Warps, Prefetch, Pipelined, MinBlocks>
      <<<n / (32 / G), Warps * 32, 0, stream>>>(codes, scales, input, output, n,
                                                k, input_stride, rounds,
                                                scale_mul, out_mul);
}

// Instantiated launch configurations (K, N, groups, warps, prefetch,
// pipelined, min_blocks); K and N only label table rows. Production builds
// only the measured table entries.
#ifdef VLLM_NVFP4_TM_GEMV_STANDALONE
  #define VLLM_TM_GEMV_SHAPES(X) \
    X(0, 0, 1, 8, 2, true, 1)    \
    X(0, 0, 1, 8, 4, true, 1)    \
    X(0, 0, 1, 16, 2, true, 2)   \
    X(0, 0, 1, 16, 4, false, 1)  \
    X(0, 0, 1, 16, 2, false, 2)  \
    X(0, 0, 2, 4, 4, true, 1)    \
    X(0, 0, 2, 8, 2, true, 1)    \
    X(0, 0, 2, 8, 2, true, 4)    \
    X(0, 0, 2, 8, 4, false, 1)   \
    X(0, 0, 2, 8, 2, false, 4)   \
    X(0, 0, 2, 16, 2, false, 2)  \
    X(0, 0, 2, 16, 1, false, 2)  \
    X(0, 0, 2, 16, 3, false, 2)  \
    X(0, 0, 2, 8, 3, false, 2)   \
    X(0, 0, 4, 2, 2, false, 16)  \
    X(0, 0, 4, 4, 1, false, 8)   \
    X(0, 0, 4, 4, 2, false, 8)   \
    X(0, 0, 4, 4, 4, false, 4)   \
    X(0, 0, 4, 8, 1, false, 4)   \
    X(0, 0, 4, 8, 2, false, 4)   \
    X(0, 0, 4, 16, 1, false, 2)  \
    X(0, 0, 4, 2, 2, true, 1)    \
    X(0, 0, 4, 4, 2, true, 1)    \
    X(0, 0, 4, 8, 3, false, 4)   \
    X(0, 0, 2, 8, 1, false, 4)   \
    X(0, 0, 2, 4, 2, false, 8)   \
    X(0, 0, 1, 8, 2, false, 4)
#else
  #define VLLM_TM_GEMV_SHAPES(X) VLLM_TM_GEMV_TABLE(X)
#endif

template <int M>
bool dispatch_tm_gemv_rows(const TmGemvConfig& c, const uint4* codes,
                           const half* scales, const half* input, half* output,
                           int n, int k, int input_stride, half2 scale_mul,
                           float out_mul, cudaStream_t stream) {
#define VLLM_TM_GEMV_CASE(K, N, G, WARPS, PREFETCH, PIPELINED, MIN_BLOCKS)    \
  if (c.groups == G && c.warps == WARPS && c.prefetch == PREFETCH &&          \
      c.pipelined == PIPELINED && c.min_blocks == MIN_BLOCKS) {               \
    launch_tm_gemv<M, G, WARPS, PREFETCH, PIPELINED, MIN_BLOCKS>(             \
        codes, scales, input, output, n, k, input_stride, scale_mul, out_mul, \
        stream);                                                              \
    return true;                                                              \
  }
  VLLM_TM_GEMV_SHAPES(VLLM_TM_GEMV_CASE)
#undef VLLM_TM_GEMV_CASE
  return false;
}

bool tm_gemv_config_valid(const TmGemvConfig& c, int64_t k, int64_t n) {
  bool instantiated = false;
#define VLLM_TM_GEMV_MATCH(K, N, G, WARPS, PREFETCH, PIPELINED, MIN_BLOCKS) \
  instantiated |= c.groups == G && c.warps == WARPS &&                      \
                  c.prefetch == PREFETCH && c.pipelined == PIPELINED &&     \
                  c.min_blocks == MIN_BLOCKS;
  VLLM_TM_GEMV_SHAPES(VLLM_TM_GEMV_MATCH)
#undef VLLM_TM_GEMV_MATCH
  if (!instantiated || k <= 0 || n <= 0 || n % 32 != 0) {
    return false;
  }
  const int64_t unit_k = 32 * c.groups;
  return k % unit_k == 0 && (k / unit_k) % (c.warps * c.prefetch) == 0;
}

// Operand contract shared by the GEMV entries. Activation rows may be
// strided (16-byte aligned); everything else is dense.
bool tm_gemv_operands_ok(const torch::Tensor& out, const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& scales) {
  if (!(out.is_cuda() && input.is_cuda() && weight.is_cuda() &&
        scales.is_cuda()) ||
      out.get_device() != input.get_device() ||
      out.get_device() != weight.get_device() ||
      out.get_device() != scales.get_device() ||
      out.scalar_type() != torch::kFloat16 ||
      input.scalar_type() != torch::kFloat16 ||
      weight.scalar_type() != torch::kInt32 ||
      scales.scalar_type() != torch::kFloat16 || out.dim() != 2 ||
      input.dim() != 2 || weight.dim() != 2 || scales.dim() != 2 ||
      !out.is_contiguous() || !weight.is_contiguous() ||
      !scales.is_contiguous()) {
    return false;
  }
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  return m >= 1 && m <= kTmGemvMaxRows && out.size(0) == m &&
         weight.size(0) == k && weight.size(1) * 8 == n &&
         scales.size(0) * 16 == k && scales.size(1) == n &&
         input.stride(1) == 1 && (m == 1 || input.stride(0) >= k) &&
         input.stride(0) % 8 == 0 &&
         reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0;
}

// Launch table measured on V100-PCIE-32GB for the Flint TP2 dense shapes
// (benchmarks/kernels/benchmark_sm70_nvfp4_m1.py: cold L2, CUDA graph
// replay). Shapes outside the table keep TurboMind.
struct TmGemvEntry {
  int k;
  int n;
  TmGemvConfig config;
};
constexpr TmGemvEntry kTmGemvTable[] = {
#define VLLM_TM_GEMV_TABLE_ENTRY(K, N, G, WARPS, PREFETCH, PIPELINED, \
                                 MIN_BLOCKS)                          \
  {K, N, {G, WARPS, PREFETCH, PIPELINED, MIN_BLOCKS}},
    VLLM_TM_GEMV_TABLE(VLLM_TM_GEMV_TABLE_ENTRY)
#undef VLLM_TM_GEMV_TABLE_ENTRY
};

TmGemvConfig tm_gemv_config(int64_t k, int64_t n) {
  for (const auto& entry : kTmGemvTable) {
    if (entry.k == k && entry.n == n &&
        tm_gemv_config_valid(entry.config, k, n)) {
      return entry.config;
    }
  }
  return TmGemvConfig{0, 0, 0, false, 0};
}

void run_tm_gemv(torch::Tensor& out, const torch::Tensor& input,
                 const torch::Tensor& weight, const torch::Tensor& scales,
                 const TmGemvConfig& config, int64_t scale_shift) {
  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(out.size(1));
  TORCH_CHECK(scale_shift >= 0 && scale_shift <= 14,
              "NVFP4 TM GEMV scale_shift must be in [0, 14]");
  // Weight = fp16(code * 2^-14 * fp16(scale * 2^shift)); the remaining
  // 2^(14 - shift) is applied in FP32. With shift 14 every weight equals
  // TurboMind's fp16(code * scale).
  const half2 scale_mul =
      __float2half2_rn(static_cast<float>(1 << scale_shift));
  const float out_mul = static_cast<float>(1 << (14 - scale_shift));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* codes =
      reinterpret_cast<const uint4*>(weight.data_ptr<int32_t>());
  const auto* scale_ptr =
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* input_ptr =
      reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* output_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const int input_stride = static_cast<int>(input.stride(0));
  const bool launched =
      m == 1 ? dispatch_tm_gemv_rows<1>(config, codes, scale_ptr, input_ptr,
                                        output_ptr, n, k, input_stride,
                                        scale_mul, out_mul, stream)
             : dispatch_tm_gemv_rows<2>(config, codes, scale_ptr, input_ptr,
                                        output_ptr, n, k, input_stride,
                                        scale_mul, out_mul, stream);
  TORCH_CHECK(launched,
              "NVFP4 TM GEMV has no kernel for groups=", config.groups,
              " warps=", config.warps, " prefetch=", config.prefetch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// True when the GEMV kernel covers this M/K/N.
bool nvfp4_tm_gemv_sm70_supported(int64_t m, int64_t k, int64_t n) {
  return m >= 1 && m <= kTmGemvMaxRows && tm_gemv_config(k, n).groups > 0;
}

#ifndef VLLM_NVFP4_TM_GEMV_STANDALONE
// M <= 2 runs the GEMV on the TurboMind operands; larger M, other shapes
// and other activations run TurboMind's nvfp4_gemm_sm70_out unchanged. The
// M branch lives here so torch.compile sees one opaque operator, and CUDA
// graphs (captured per batch size) record exactly one of the two kernels.
void nvfp4_tm_gemv_dispatch_sm70_out(torch::Tensor out, torch::Tensor input,
                                     torch::Tensor weight, torch::Tensor scales,
                                     int64_t group_size, int64_t k_ld,
                                     int64_t q_ld, int64_t scale_shift) {
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = out.size(1);
  if (group_size == 16 && k_ld == 32 * k && q_ld == n &&
      nvfp4_tm_gemv_sm70_supported(m, k, n) &&
      tm_gemv_operands_ok(out, input, weight, scales)) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    run_tm_gemv(out, input, weight, scales, tm_gemv_config(k, n), scale_shift);
    return;
  }
  nvfp4_gemm_sm70_out(out, input, weight, scales, group_size, k_ld, q_ld,
                      false);
}
#endif

#ifdef VLLM_NVFP4_TM_GEMV_STANDALONE
// Tuning entries for a task-local extension (no vLLM rebuild).
void nvfp4_tm_gemv_sm70_tune_out(torch::Tensor out, torch::Tensor input,
                                 torch::Tensor weight, torch::Tensor scales,
                                 int64_t groups, int64_t warps,
                                 int64_t prefetch, bool pipelined,
                                 int64_t min_blocks, int64_t scale_shift) {
  TORCH_CHECK(tm_gemv_operands_ok(out, input, weight, scales),
              "TM GEMV operand mismatch");
  const TmGemvConfig config{static_cast<int>(groups), static_cast<int>(warps),
                            static_cast<int>(prefetch), pipelined,
                            static_cast<int>(min_blocks)};
  TORCH_CHECK(tm_gemv_config_valid(config, input.size(1), out.size(1)),
              "invalid TM GEMV tuning configuration");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  run_tm_gemv(out, input, weight, scales, config, scale_shift);
}

TORCH_LIBRARY_FRAGMENT(_tm_gemv_dev, ops) {
  ops.def(
      "tune(Tensor(a!) out, Tensor input, Tensor weight, Tensor scales, "
      "int groups, int warps, int prefetch, bool pipelined, int min_blocks, "
      "int scale_shift) -> ()");
  ops.impl("tune", torch::kCUDA, &nvfp4_tm_gemv_sm70_tune_out);
}
#endif
