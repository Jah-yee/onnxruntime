// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tensor-core (mma.sync.m16n8k16, BF16 inputs, FP32 accumulation) INT2 block-wise (group size 32, 64, or 128)
// MoE decode/prefill GEMV.
//
// In SM80 prepacked INT2 (`kInterleave = 8`, `kTileSize = 64`), an interleaved weight row holds 8 real columns
// (`8 * k / 4 = 2k` bytes). Inside each 64-wide K tile (`128` bytes across the 8 columns), column `c` (`0..7`)
// is a contiguous 16-byte `uint4` (64 2-bit codes). Lane `(g, t)` loads one 64-wide K tile `kt64 = (step0 + s) * 4 + t`
// per step (`256` K elements per warp step): `we` (16B for even column `2*g`) and `wo` (16B for odd column `2*g + 1`),
// matched with the 64 BF16 activations `b_lo[0..3]` and `b_hi[0..3]` of route `g`.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace onnxruntime::llm::kernels::fpA_intB_gemv {
namespace moe_tc {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRoutes = 8;  // mma N
constexpr int kMmaM = 16;      // output columns per m-block

struct Params {
  const __nv_bfloat16* act;
  const uint8_t* weight;
  const __nv_bfloat16* scales;
  const __nv_bfloat16* bias;
  __nv_bfloat16* out;
  const int64_t* expert_first_token_offset;
  const int* permuted_row_to_expert;
  const int* permuted_row_to_source_row;
  const float* swiglu_alpha;
  const float* swiglu_beta;
  const float* swiglu_limit;
  float alpha;
  float beta;
  float limit;
  int64_t weight_expert_stride;  // bytes
  int64_t scale_expert_stride;   // elements
  int num_experts;
  int n;  // real output columns (2 * inter_size for the interleaved SwiGLU kernel)
  int k;
  int num_rows;
  int expanded_num_rows;
};

#if defined(__CUDACC__)

__device__ __forceinline__ uint4 LoadWeightsStreaming(const void* ptr) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

template <int J>
__device__ __forceinline__ uint32_t Comp(const uint4& v) {
  if constexpr (J == 0) {
    return v.x;
  } else if constexpr (J == 1) {
    return v.y;
  } else if constexpr (J == 2) {
    return v.z;
  } else {
    return v.w;
  }
}

__device__ __forceinline__ uint32_t Bf16x2Mul(uint32_t a, uint32_t b) {
  uint32_t d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
#else
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
#endif
  return d;
}

__device__ __forceinline__ void MmaBf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// MaskIdx in {0, 1, 2} selects 2-bit fields at bits {0..1, 2..3, 4..5} in both 16-bit halves of `x`,
// all of which fit inside BF16's 7-bit mantissa (0x007F007F).
template <int MaskIdx>
__device__ __forceinline__ uint32_t DequantPairInt2Masked(uint32_t x, uint32_t scale2) {
  static_assert(MaskIdx >= 0 && MaskIdx < 3);
  constexpr uint32_t kMask = (MaskIdx == 0) ? 0x00030003u : ((MaskIdx == 1) ? 0x000C000Cu : 0x00300030u);
  constexpr uint32_t kMul = (MaskIdx == 0) ? 0x3F803F80u : ((MaskIdx == 1) ? 0x3E803E80u : 0x3D803D80u);
  constexpr uint32_t kAdd = (MaskIdx == 0) ? 0xC302C302u : ((MaskIdx == 1) ? 0xC208C208u : 0xC120C120u);
  uint32_t v;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(v)
      : "r"(x), "n"(kMask), "n"(0x43004300u), "n"((0xf0 & 0xcc) | 0xaa));
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(v), "r"(kMul), "r"(kAdd));
  return Bf16x2Mul(r, scale2);
}

// One 32-bit weight word of a 64-K SM80 INT2 tile holds 16 2-bit codes (8 bf16x2 pairs: 4 in [0..31] and 4 in [32..63])
// feeds four MMA steps: word J holds logical pairs q = J + 4*p (p = 0..7), matched with comp J of b_lo[0..3] and b_hi[0..3].
template <int J, int kM>
__device__ __forceinline__ void ComputeWordInt2(float (&acc)[kM][4], int mb, const uint4& we, const uint4& wo,
                                                uint32_t se_lo, uint32_t so_lo, uint32_t se_hi, uint32_t so_hi,
                                                const uint4 (&b_lo)[4], const uint4 (&b_hi)[4]) {
  const uint32_t ew0 = Comp<J>(we);
  const uint32_t ow0 = Comp<J>(wo);
  const uint32_t ew6 = ew0 >> 6;
  const uint32_t ow6 = ow0 >> 6;
  const uint32_t ew12 = ew0 >> 12;
  const uint32_t ow12 = ow0 >> 12;

  const uint32_t e0 = DequantPairInt2Masked<0>(ew0, se_lo);
  const uint32_t o0 = DequantPairInt2Masked<0>(ow0, so_lo);
  const uint32_t e1 = DequantPairInt2Masked<1>(ew0, se_lo);
  const uint32_t o1 = DequantPairInt2Masked<1>(ow0, so_lo);
  MmaBf16(acc[mb], e0, o0, e1, o1, Comp<J>(b_lo[0]), Comp<J>(b_lo[1]));

  const uint32_t e2 = DequantPairInt2Masked<2>(ew0, se_lo);
  const uint32_t o2 = DequantPairInt2Masked<2>(ow0, so_lo);
  const uint32_t e3 = DequantPairInt2Masked<0>(ew6, se_lo);
  const uint32_t o3 = DequantPairInt2Masked<0>(ow6, so_lo);
  MmaBf16(acc[mb], e2, o2, e3, o3, Comp<J>(b_lo[2]), Comp<J>(b_lo[3]));

  const uint32_t e4 = DequantPairInt2Masked<1>(ew6, se_hi);
  const uint32_t o4 = DequantPairInt2Masked<1>(ow6, so_hi);
  const uint32_t e5 = DequantPairInt2Masked<2>(ew6, se_hi);
  const uint32_t o5 = DequantPairInt2Masked<2>(ow6, so_hi);
  MmaBf16(acc[mb], e4, o4, e5, o5, Comp<J>(b_hi[0]), Comp<J>(b_hi[1]));

  const uint32_t e6 = DequantPairInt2Masked<0>(ew12, se_hi);
  const uint32_t o6 = DequantPairInt2Masked<0>(ow12, so_hi);
  const uint32_t e7 = DequantPairInt2Masked<1>(ew12, se_hi);
  const uint32_t o7 = DequantPairInt2Masked<1>(ow12, so_hi);
  MmaBf16(acc[mb], e6, o6, e7, o7, Comp<J>(b_hi[2]), Comp<J>(b_hi[3]));
}

__device__ __forceinline__ bool DiscoverExpertRouteChunk(const Params& p, int row0, int expert, int lane,
                                                         int& routes, uint64_t& members, bool& contiguous_routes) {
  members = 0;
  routes = 0;
  if (p.expert_first_token_offset != nullptr) {
    const int64_t first = p.expert_first_token_offset[expert];
    const int64_t last = p.expert_first_token_offset[expert + 1];
    if (((row0 - first) & (kMaxRoutes - 1)) != 0) {
      return false;
    }
    routes = (last - row0) < kMaxRoutes ? static_cast<int>(last - row0) : kMaxRoutes;
    contiguous_routes = true;
    return routes > 0;
  }
  const int total = p.expanded_num_rows;
  if (total <= 256) {
    int ev[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = i * 32 + lane;
      ev[i] = idx < total ? __ldg(p.permuted_row_to_expert + idx) : -1;
    }
    const int c0 = row0 >> 5;
    const int b0 = row0 & 31;
    uint32_t match[8];
    int rank = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      match[i] = __ballot_sync(0xffffffffu, ev[i] == expert);
      const uint32_t below = i < c0 ? 0xffffffffu : (i == c0 ? ((1u << b0) - 1u) : 0u);
      rank += __popc(match[i] & below);
    }
    if ((rank & (kMaxRoutes - 1)) != 0) {
      return false;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t bits = match[i] & (i < c0 ? 0u : (i == c0 ? (0xffffffffu << b0) : 0xffffffffu));
      while (bits != 0u && routes < kMaxRoutes) {
        const int pos = (i << 5) + (__ffs(bits) - 1);
        bits &= bits - 1u;
        members |= static_cast<uint64_t>(pos) << (8 * routes);
        ++routes;
      }
    }
    contiguous_routes = false;
    return routes > 0;
  }
  int first = row0;
  if (row0 > 0 && __ldg(&p.permuted_row_to_expert[row0 - 1]) == expert) {
    int lo = 0;
    int hi = row0 - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (__ldg(&p.permuted_row_to_expert[mid]) < expert) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    first = lo;
  }
  if (((row0 - first) & (kMaxRoutes - 1)) != 0) {
    return false;
  }
  const bool same = (lane < kMaxRoutes) && ((row0 + lane) < total) &&
                    (__ldg(&p.permuted_row_to_expert[row0 + lane]) == expert);
  routes = __popc(__ballot_sync(0xffffffffu, same));
  contiguous_routes = true;
  return routes > 0;
}

constexpr int MinCtasPerSm(int kM) {
  return kM == 1 ? 6 : 4;
}

template <bool kSwiglu, int kWarpsK, int kM, int kGroupSize>
__global__ void __launch_bounds__(kThreads, MinCtasPerSm(kM)) MoeGemvTcInt2Kernel(const Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  static_assert(kWarps % kWarpsK == 0, "warps must split evenly over K");
  static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128, "unsupported INT2 group size");
  constexpr int kWarpsN = kWarps / kWarpsK;
  constexpr int kTileN = kWarpsN * kM * kMmaM;

  const int row0 = blockIdx.x;
  const int expert = __ldg(&p.permuted_row_to_expert[row0]);
  if (expert < 0 || expert >= p.num_experts) {
    return;
  }

  const int lane = threadIdx.x & 31;
  uint64_t members = 0;
  int routes = 0;
  bool contiguous_routes = false;
  if (!DiscoverExpertRouteChunk(p, row0, expert, lane, routes, members, contiguous_routes)) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int wk = warp % kWarpsK;
  const int wn = warp / kWarpsK;
  const int n = p.n;
  const int k = p.k;
  const int col_w = blockIdx.y * kTileN + wn * (kM * kMmaM);
  const int total_tiles64 = k / 64;
  const int total_steps = (total_tiles64 + 3) / 4;
  const int steps = total_steps / kWarpsK;
  const int step0 = wk * steps;

  const int64_t row_bytes = 2 * static_cast<int64_t>(k);
  const uint8_t* w_base = p.weight + static_cast<int64_t>(expert) * p.weight_expert_stride +
                          (static_cast<int64_t>(col_w / 8) + (g >> 2)) * row_bytes + (g & 3) * 32;
  const __nv_bfloat16* s_base = p.scales + static_cast<int64_t>(expert) * p.scale_expert_stride + col_w + 2 * g;

  const int g_act = min(g, max(routes - 1, 0));
  const int route_g = contiguous_routes
                          ? (row0 + g_act)
                          : static_cast<int>((members >> (8 * g_act)) & 0xffu);
  int src_row = route_g;
  if constexpr (kSwiglu) {
    if (p.permuted_row_to_source_row != nullptr) {
      src_row = p.permuted_row_to_source_row[route_g] % p.num_rows;
    }
  }
  const __nv_bfloat16* a_row = p.act + static_cast<int64_t>(src_row) * k;

  float acc[kM][4];
#pragma unroll
  for (int mb = 0; mb < kM; ++mb) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[mb][i] = 0.f;
    }
  }

  auto load_step_acts = [&](int step_idx, uint4(&dst_lo)[4], uint4(&dst_hi)[4]) {
    const int kt64 = step_idx * 4 + t;
    if (kt64 < total_tiles64) {
      const uint4* a4 = reinterpret_cast<const uint4*>(a_row + static_cast<int64_t>(kt64) * 64);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst_lo[i] = __ldg(&a4[i]);
        dst_hi[i] = __ldg(&a4[4 + i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst_lo[i] = make_uint4(0, 0, 0, 0);
        dst_hi[i] = make_uint4(0, 0, 0, 0);
      }
    }
  };

  auto load_step_weights = [&](int step_idx, int mb, uint4& e, uint4& o, uint32_t& c_lo, uint32_t& c_hi) {
    const int kt64 = step_idx * 4 + t;
    const bool valid_kt = (kt64 < total_tiles64);
    const int kt64_safe = valid_kt ? kt64 : (total_tiles64 - 1);
    const uint8_t* w = w_base + static_cast<int64_t>(mb) * 2 * row_bytes + static_cast<int64_t>(kt64_safe) * 128;
    e = LoadWeightsStreaming(w);
    o = LoadWeightsStreaming(w + 16);
    c_lo = 0u;
    c_hi = 0u;
    if (valid_kt) {
      if constexpr (kGroupSize == 32) {
        const __nv_bfloat16* sp = s_base + static_cast<int64_t>(2 * kt64_safe) * n + mb * kMmaM;
        c_lo = __ldg(reinterpret_cast<const uint32_t*>(sp));
        c_hi = __ldg(reinterpret_cast<const uint32_t*>(sp + n));
      } else if constexpr (kGroupSize == 64) {
        c_lo = __ldg(reinterpret_cast<const uint32_t*>(s_base + static_cast<int64_t>(kt64_safe) * n + mb * kMmaM));
        c_hi = c_lo;
      } else {
        c_lo = __ldg(reinterpret_cast<const uint32_t*>(s_base + static_cast<int64_t>(kt64_safe >> 1) * n + mb * kMmaM));
        c_hi = c_lo;
      }
    }
  };

  auto compute_mb = [&](int mb, const uint4& we_cur, const uint4& wo_cur, uint32_t c_lo, uint32_t c_hi,
                        const uint4(&b_lo_cur)[4], const uint4(&b_hi_cur)[4]) {
    const uint32_t se_lo = __byte_perm(c_lo, 0, 0x1010);
    const uint32_t so_lo = __byte_perm(c_lo, 0, 0x3232);
    const uint32_t se_hi = (kGroupSize == 32) ? __byte_perm(c_hi, 0, 0x1010) : se_lo;
    const uint32_t so_hi = (kGroupSize == 32) ? __byte_perm(c_hi, 0, 0x3232) : so_lo;
    ComputeWordInt2<0, kM>(acc, mb, we_cur, wo_cur, se_lo, so_lo, se_hi, so_hi, b_lo_cur, b_hi_cur);
    ComputeWordInt2<1, kM>(acc, mb, we_cur, wo_cur, se_lo, so_lo, se_hi, so_hi, b_lo_cur, b_hi_cur);
    ComputeWordInt2<2, kM>(acc, mb, we_cur, wo_cur, se_lo, so_lo, se_hi, so_hi, b_lo_cur, b_hi_cur);
    ComputeWordInt2<3, kM>(acc, mb, we_cur, wo_cur, se_lo, so_lo, se_hi, so_hi, b_lo_cur, b_hi_cur);
  };

  uint4 we[kM], wo[kM];
  uint32_t sc_lo[kM], sc_hi[kM];
  uint4 b_lo[4], b_hi[4];
#pragma unroll
  for (int mb = 0; mb < kM; ++mb) {
    load_step_weights(step0, mb, we[mb], wo[mb], sc_lo[mb], sc_hi[mb]);
  }
  load_step_acts(step0, b_lo, b_hi);

#pragma unroll 1
  for (int s = 0; s < steps; ++s) {
    const bool more = (s + 1) < steps;
    const int next_step = more ? (step0 + s + 1) : (step0 + s);
    uint4 bn_lo[4], bn_hi[4];
    load_step_acts(next_step, bn_lo, bn_hi);
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      compute_mb(mb, we[mb], wo[mb], sc_lo[mb], sc_hi[mb], b_lo, b_hi);
      if (more) {
        load_step_weights(next_step, mb, we[mb], wo[mb], sc_lo[mb], sc_hi[mb]);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b_lo[i] = bn_lo[i];
      b_hi[i] = bn_hi[i];
    }
  }

  const __nv_bfloat16* bias = p.bias != nullptr ? p.bias + static_cast<int64_t>(expert) * n : nullptr;
  float alpha = p.alpha;
  float beta = p.beta;
  float limit = p.limit;
  if constexpr (kSwiglu) {
    alpha = p.swiglu_alpha != nullptr ? p.swiglu_alpha[expert] : alpha;
    beta = p.swiglu_beta != nullptr ? p.swiglu_beta[expert] : beta;
    limit = p.swiglu_limit != nullptr ? p.swiglu_limit[expert] : limit;
  }

  auto finalize = [&](const float4 d, int wn2, int mb, int l) {
    const int g2 = l >> 2;
    const int t2 = l & 3;
    const int col = blockIdx.y * kTileN + (wn2 * kM + mb) * kMmaM + 2 * g2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int route = 2 * t2 + h;
      if (route >= routes) {
        continue;
      }
      const int out_row = contiguous_routes
                              ? (row0 + route)
                              : static_cast<int>((members >> (8 * route)) & 0xffu);
      float even = h == 0 ? d.x : d.y;
      float odd = h == 0 ? d.z : d.w;
      if (bias != nullptr) {
        even += __bfloat162float(bias[col]);
        odd += __bfloat162float(bias[col + 1]);
      }
      if constexpr (kSwiglu) {
        float gate = even;
        float linear = odd;
        if (isfinite(limit)) {
          gate = fminf(gate, limit);
          linear = fminf(fmaxf(linear, -limit), limit);
        }
        linear += beta;
        const float sigmoid = 1.0f / (1.0f + expf(-alpha * gate));
        p.out[static_cast<int64_t>(out_row) * (n / 2) + col / 2] =
            __float2bfloat16_rn(gate * sigmoid * linear);
      } else {
        *reinterpret_cast<__nv_bfloat162*>(p.out + static_cast<int64_t>(out_row) * n + col) =
            __floats2bfloat162_rn(even, odd);
      }
    }
  };

  if constexpr (kWarpsK > 1) {
    __shared__ float4 red[kWarpsK][kWarpsN][kM][32];
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      red[wk][wn][mb][lane] = make_float4(acc[mb][0], acc[mb][1], acc[mb][2], acc[mb][3]);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < kWarpsN * kM * 32; idx += kThreads) {
      const int l = idx & 31;
      const int mb = (idx >> 5) % kM;
      const int wn2 = (idx >> 5) / kM;
      float4 d = red[0][wn2][mb][l];
#pragma unroll
      for (int w = 1; w < kWarpsK; ++w) {
        const float4 e = red[w][wn2][mb][l];
        d.x += e.x;
        d.y += e.y;
        d.z += e.z;
        d.w += e.w;
      }
      finalize(d, wn2, mb, l);
    }
  } else {
#pragma unroll
    for (int mb = 0; mb < kM; ++mb) {
      finalize(make_float4(acc[mb][0], acc[mb][1], acc[mb][2], acc[mb][3]), wn, mb, lane);
    }
  }
#endif
}

inline bool DeviceSupportsBf16Mma() {
  int device = 0;
  int major = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
    return false;
  }
  return major >= 8;
}

template <bool kSwiglu, int kWarpsK, int kM, int kGroupSize>
inline void LaunchInt2(const Params& p, int64_t expanded_num_rows, cudaStream_t stream) {
  constexpr int kTileN = (kWarps / kWarpsK) * kM * kMmaM;
  const dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(p.n / kTileN));
  MoeGemvTcInt2Kernel<kSwiglu, kWarpsK, kM, kGroupSize><<<grid, kThreads, 0, stream>>>(p);
}

inline bool TryLaunchInt2Bf16(bool swiglu, const __nv_bfloat16* act, const uint8_t* weight,
                              const __nv_bfloat16* scales, const __nv_bfloat16* bias, __nv_bfloat16* out,
                              const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                              int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                              const float* swiglu_alpha, const float* swiglu_beta, const float* swiglu_limit,
                              float alpha, float beta, float limit, const int* permuted_row_to_source_row,
                              int64_t num_rows, cudaStream_t stream, int tensor_core_min_rows) {
  if ((group_size != 32 && group_size != 64 && group_size != 128) || permuted_row_to_expert == nullptr ||
      expanded_num_rows < tensor_core_min_rows || tensor_core_min_rows == 0 || !DeviceSupportsBf16Mma()) {
    return false;
  }
  if (k % 64 != 0 || k % group_size != 0 || n % 64 != 0) {
    return false;
  }
  const int total_steps = ((static_cast<int>(k) / 64) + 3) / 4;
  const bool use_2warps_k = swiglu && (total_steps % 2 == 0) && (n % 64 == 0);

  cudaFuncAttributes attributes{};
  const auto check_kernel = group_size == 64   ? (use_2warps_k ? MoeGemvTcInt2Kernel<true, 2, 2, 64>
                                                               : MoeGemvTcInt2Kernel<false, 1, 1, 64>)
                            : group_size == 32 ? (use_2warps_k ? MoeGemvTcInt2Kernel<true, 2, 2, 32>
                                                               : MoeGemvTcInt2Kernel<false, 1, 1, 32>)
                                               : (use_2warps_k ? MoeGemvTcInt2Kernel<true, 2, 2, 128>
                                                               : MoeGemvTcInt2Kernel<false, 1, 1, 128>);
  if (cudaFuncGetAttributes(&attributes, check_kernel) != cudaSuccess || attributes.ptxVersion < 80) {
    return false;
  }

  Params p;
  p.act = act;
  p.weight = weight;
  p.scales = scales;
  p.bias = bias;
  p.out = out;
  p.expert_first_token_offset = expert_first_token_offset;
  p.permuted_row_to_expert = permuted_row_to_expert;
  p.permuted_row_to_source_row = permuted_row_to_source_row;
  p.swiglu_alpha = swiglu_alpha;
  p.swiglu_beta = swiglu_beta;
  p.swiglu_limit = swiglu_limit;
  p.alpha = alpha;
  p.beta = beta;
  p.limit = limit;
  p.weight_expert_stride = n * k / 4;
  p.scale_expert_stride = (k / group_size) * n;
  p.num_experts = num_experts;
  p.n = static_cast<int>(n);
  p.k = static_cast<int>(k);
  p.num_rows = static_cast<int>(num_rows);
  p.expanded_num_rows = static_cast<int>(expanded_num_rows);

  auto dispatch_group = [&](auto group_tag) {
    constexpr int kG = decltype(group_tag)::value;
    if (swiglu) {
      if (use_2warps_k) {
        LaunchInt2<true, 2, 2, kG>(p, expanded_num_rows, stream);
      } else {
        LaunchInt2<true, 1, 1, kG>(p, expanded_num_rows, stream);
      }
    } else {
      LaunchInt2<false, 1, 1, kG>(p, expanded_num_rows, stream);
    }
  };
  if (group_size == 64) {
    dispatch_group(std::integral_constant<int, 64>{});
  } else if (group_size == 32) {
    dispatch_group(std::integral_constant<int, 32>{});
  } else {
    dispatch_group(std::integral_constant<int, 128>{});
  }
  return true;
}

#endif  // __CUDACC__

}  // namespace moe_tc
}  // namespace onnxruntime::llm::kernels::fpA_intB_gemv
