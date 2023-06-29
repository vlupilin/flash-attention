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
  explicit FmhaBwdRunner(const LaunchParams<FmhaDgradParams> &launch_params)
    : params(launch_params.params),
      is_bf16(launch_params.is_bf16),
      is_casual(launch_params.is_casual),
      is_performance_mode(launch_params.is_performance_mode),
      is_deterministic(launch_params.is_deterministic) {}
  // run fmha bwd
  void Run();
  // destructor
  ~FmhaBwdRunner() = default;
 
 private:
  // Todo: pass as a pointer
  FmhaDgradParams &params;
  bool is_bf16;
  bool is_casual;
  bool is_performance_mode;
  bool is_deterministic;
} // class FmhaBwdRunner
} // namespace bwd_device_gemm