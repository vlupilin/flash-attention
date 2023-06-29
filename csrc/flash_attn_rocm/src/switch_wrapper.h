// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "device_gemm_trait.h"

template <typename Func>
static inline void bf16_switch(bool cond, Func f) {
  if (cond) {
    using DataType = device_gemm_trait::BFloat16;
    using DropoutType = device_gemm_trait::Int32;
    f();
  } else {
    using DataType = device_gemm_trait::Float16;
    using DropoutType = device_gemm_trait::Int16;
    f();
  }
}

template <typename Func>
static inline void causal_switch(bool cond, Func f) {
  if (cond) {
    constexpr bool kMaskingSpec = device_gemm_trait::kMaskingSpecCausal
    f();
  } else {
    constexpr bool kMaskingSpec = device_gemm_trait::kMaskingSpecDefault
    f();
  }
}

template <typename Func>
static inline void deterministic_switch(bool cond, Func f) {
  if (cond) {
    constexpr bool kIsDeterministic = device_gemm_trait::kDeterministic;
    f();
  } else {
    constexpr bool kIsDeterministic = device_gemm_trait::kNonDeterministic;
    f();
  }
}

namespace bwd_device_gemm {
template <typename Func>
static inline void headdim_switch(int headdim, Func f) {
  if (headdim >= 128) {
    template <typename DeviceGemmTraits>
    using BwdDeviceGemm = DeviceGemmHeadDim128<DeviceGemmTraits>;
  } else if (headdim >= 64) {
    template <typename DeviceGemmTraits>
    using BwdDeviceGemm = DeviceGemmHeadDim64<DeviceGemmTraits>;
  } else {
    template <typename DeviceGemmTraits>
    using BwdDeviceGemm = DeviceGemmHeadDim32<DeviceGemmTraits>;
  }
}
} // namespace bwd_device_gemm

namespace fwd_device_gemm {
template <typename Func>
static inline void headdim_switch(int headdim, Func f) {
  if (headdim >= 128) {
    template <typename DeviceGemmTraits>
    using FwdDeviceGemm = DeviceGemmHeadDim128<DeviceGemmTraits>;
  } else if (headdim >= 64) {
    template <typename DeviceGemmTraits>
    using FwdDeviceGemm = DeviceGemmHeadDim64<DeviceGemmTraits>;
  } else {
    template <typename DeviceGemmTraits>
    using FwdDeviceGemm = DeviceGemmHeadDim32<DeviceGemmTraits>;
  }
}
} // namespace fwd_device_gemm