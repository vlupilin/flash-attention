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

#include <memory>

#include "fwd_device_gemm_template.h"
#include "device_gemm_trait.h"
#include "launch_params.h"

namespace fwd_device_gemm {
template <template <typename> typename DeviceGemmTemplate, typename DeviceGemmTraits>
class FwdDeviceGemmInstanceLauncherBase {
 public:
  // constructor
  explicit FwdDeviceGemmInstanceLauncherBase()
    : device_gemm_instance_ptr_(std::make_unique<DeviceGemmTemplate<DeviceGemmTraits>>()) {}
  
  virtual void Launch(const FmhaFwdParams &params) = 0;

 protected:
  std::unique_ptr<DeviceGemmTemplate<DeviceGemmTraits>> device_gemm_instance_ptr_;
}; // class FwdDeviceGemmInstanceLauncher

template <typename DeviceGemmTraits>
class FwdDeviceGemmInstanceLauncherHeadDim32
    : public FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim32, DeviceGemmTraits> {
 public:
  // constructor
  explicit FwdDeviceGemmInstanceLauncherHeadDim32()
    : FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim32, DeviceGemmTraits>() {}

  void Launch(const FmhaFwdParams &params) override;
}; // class FwdDeviceGemmInstanceLauncherHeadDim32

template <typename DeviceGemmTraits>
class FwdDeviceGemmInstanceLauncherHeadDim64
    : public FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim64, DeviceGemmTraits> {
 public:
  // constructor
  explicit FwdDeviceGemmInstanceLauncherHeadDim64()
    : FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim64, DeviceGemmTraits>() {}

  void Launch(const FmhaFwdParams &params) override;
}; // class FwdDeviceGemmInstanceLauncherHeadDim64

template <typename DeviceGemmTraits>
class FwdDeviceGemmInstanceLauncherHeadDim128
    : public FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim128, DeviceGemmTraits> {
 public:
  // constructor
  explicit FwdDeviceGemmInstanceLauncherHeadDim128()
    : FwdDeviceGemmInstanceLauncherBase<DeviceGemmHeadDim128, DeviceGemmTraits>() {}

  void Launch(const FmhaFwdParams &params) override;
}; // class FwdDeviceGemmInstanceLauncherHeadDim128
} // namespace fwd_device_gemm