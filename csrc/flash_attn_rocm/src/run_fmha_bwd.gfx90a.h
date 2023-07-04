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

#include "bwd_device_gemm_launcher.h"
#include "launch_params.h"

namespace bwd_device_gemm {
class FmhaBwdRunner {
 public:
  // constructor
  explicit FmhaBwdRunner(const LaunchParams<FmhaBwdParams> &launch_params)
    : params_(launch_params.params),
      is_bf16_(launch_params.is_bf16_),
      is_causal_(launch_params.is_causal_),
      is_deterministic_(launch_params.is_deterministic_),
      is_performance_mode_(launch_params.is_performance_mode_) {}
  // run fmha bwd
  void Run();
 
 protected:
  const FmhaBwdParams &params_;
  const bool is_bf16_;
  const bool is_causal_;
  const bool is_deterministic_;
  const bool is_performance_mode_;

  template <typename DataType, typename DropoutType, bool kIsDeterministic, bool kMaskingSpec>
  void run_() {
    if (is_performance_mode_) {
      // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
      using BwdDeviceGemmTraits = device_gemm_trait::Backward<DataType, DataType, device_gemm_trait::BFloat16, DropoutType, 4, kMaskingSpec, kIsDeterministic>;
      using BwdDeviceGemmTemplate = BwdDeviceGemm<BwdDeviceGemmTraits>;
      auto bwd_device_gemm_instance_launcher_ptr = std::make_unique<BwdDeviceGemmInstanceLauncher<BwdDeviceGemmTemplate>>();
      bwd_device_gemm_instance_launcher_ptr->Launch(params_);
    }
    // non-performance mode for unit test
    else {
      // input, output, gemm, dropout, cshuffle, masking specialization, deterministic
      using BwdDeviceGemmTraits = device_gemm_trait::Backward<DataType, device_gemm_trait::Float32, DataType, DropoutType, 4, kMaskingSpec, kIsDeterministic>;
      using BwdDeviceGemmTemplate = BwdDeviceGemm<BwdDeviceGemmTraits>;
      auto bwd_device_gemm_instance_launcher_ptr = std::make_unique<BwdDeviceGemmInstanceLauncher<BwdDeviceGemmTemplate>>();
      bwd_device_gemm_instance_launcher_ptr->Launch(params_);
    }
  }
}; // class FmhaBwdRunner
} // namespace bwd_device_gemm