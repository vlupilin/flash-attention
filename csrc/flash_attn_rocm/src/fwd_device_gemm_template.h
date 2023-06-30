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

namespace fwd_device_gemm {
namespace device_op = ck::tensor_operation::device; // namespace alias for internal use
// type alias for DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle with head_dim = 32
template <typename FwdDeviceGemmTraits>
using DeviceGemmHeadDim32 = device_op::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle<
        FwdDeviceGemmTraits::kNumDimG,
        FwdDeviceGemmTraits::kNumDimM, 
        FwdDeviceGemmTraits::kNumDimN, 
        FwdDeviceGemmTraits::kNumDimK, 
        FwdDeviceGemmTraits::kNumDimO, 
        typename FwdDeviceGemmTraits::ADataType,
        typename FwdDeviceGemmTraits::B0DataType,
        typename FwdDeviceGemmTraits::B1DataType,
        typename FwdDeviceGemmTraits::CDataType,
        typename FwdDeviceGemmTraits::GemmDataType,
        typename FwdDeviceGemmTraits::ZDataType,
        typename FwdDeviceGemmTraits::LSEDataType,
        typename FwdDeviceGemmTraits::Acc0BiasDataType,
        typename FwdDeviceGemmTraits::Acc1BiasDataType,
        typename FwdDeviceGemmTraits::AccDataType,
        typename FwdDeviceGemmTraits::CShuffleDataType,
        typename FwdDeviceGemmTraits::AElementOp,
        typename FwdDeviceGemmTraits::B0ElementOp,
        typename FwdDeviceGemmTraits::Acc0ElementOp,
        typename FwdDeviceGemmTraits::B1ElementOp,
        typename FwdDeviceGemmTraits::CElementOp,
        FwdDeviceGemmTraits::kGemmSpec,
        FwdDeviceGemmTraits::kTensorSpecA,
        FwdDeviceGemmTraits::kTensorSpecB0,
        FwdDeviceGemmTraits::kTensorSpecB1,
        FwdDeviceGemmTraits::kTensorSpecC,
        1,
        256,
        128,   // MPerBlock
        128,   // NPerBlock
        32,    // KPerBlock
        32,    // Gemm1NPerBlock
        32,    // Gemm1KPerBlock
        8,     // AK1
        8,     // BK1
        2,     // B1K1
        32,           // MPerXDL
        32,           // NPerXDL
        1,            // MXdlPerWave
        4,       // NXdlPerWave
        1,  // Gemm1NXdlPerWave
        device_gemm_trait::S<4, 64, 1>,    // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,   // ABlockLdsExtraM
        device_gemm_trait::S<4, 64, 1>,    // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,  // B0BlockLdsExtraN
        device_gemm_trait::S<16, 16, 1>,   // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>,
        device_gemm_trait::S<0, 2, 1>,
        1,
        2,    //B1BlockTransferSrcScalarPerVector
        2,
        false,
        1,        // CShuffleMXdlPerWavePerShuffle
        1,        // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 64, 1, 4>,  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,                                    // CShuffleBlockTransferScalarPerVector_NPerBlock
        FwdDeviceGemmTraits::kMaskingSpec,                      // MaskingSpecialization
        FwdDeviceGemmTraits::kIsDeterministic>;
// type alias for DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle with head_dim = 64
template <typename FwdDeviceGemmTraits>
using DeviceGemmHeadDim64 = device_op::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle<
        FwdDeviceGemmTraits::kNumDimG, 
        FwdDeviceGemmTraits::kNumDimM, 
        FwdDeviceGemmTraits::kNumDimN, 
        FwdDeviceGemmTraits::kNumDimK, 
        FwdDeviceGemmTraits::kNumDimO, 
        typename FwdDeviceGemmTraits::ADataType,
        typename FwdDeviceGemmTraits::B0DataType,
        typename FwdDeviceGemmTraits::B1DataType,
        typename FwdDeviceGemmTraits::CDataType,
        typename FwdDeviceGemmTraits::GemmDataType,
        typename FwdDeviceGemmTraits::ZDataType,
        typename FwdDeviceGemmTraits::LSEDataType,
        typename FwdDeviceGemmTraits::Acc0BiasDataType,
        typename FwdDeviceGemmTraits::Acc1BiasDataType,
        typename FwdDeviceGemmTraits::AccDataType,
        typename FwdDeviceGemmTraits::CShuffleDataType,
        typename FwdDeviceGemmTraits::AElementOp,
        typename FwdDeviceGemmTraits::B0ElementOp,
        typename FwdDeviceGemmTraits::Acc0ElementOp,
        typename FwdDeviceGemmTraits::B1ElementOp,
        typename FwdDeviceGemmTraits::CElementOp,
        FwdDeviceGemmTraits::kGemmSpec,
        FwdDeviceGemmTraits::kTensorSpecA,
        FwdDeviceGemmTraits::kTensorSpecB0,
        FwdDeviceGemmTraits::kTensorSpecB1,
        FwdDeviceGemmTraits::kTensorSpecC,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,         // KPerBlock
        64,    // Gemm1NPerBlock
        32,    // Gemm1KPerBlock
        8,                 // AK1
        8,                 // BK1
        2,                 // B1K1
        32,           // MPerXDL
        32,           // NPerXDL
        1,                 // MXdlPerWave
        4,       // NXdlPerWave
        2,  // Gemm1NXdlPerWave
        device_gemm_trait::S<4, 64, 1>,    // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,   // ABlockLdsExtraM
        device_gemm_trait::S<4, 64, 1>,    // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,  // B0BlockLdsExtraN
        device_gemm_trait::S<16, 16, 1>,   // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>,
        device_gemm_trait::S<0, 2, 1>,
        1,
        4,    //B1BlockTransferSrcScalarPerVector
        2,
        false,
        1,                                    // CShuffleMXdlPerWavePerShuffle
        2,        // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 32, 1, 8>,  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,                                    // CShuffleBlockTransferScalarPerVector_NPerBlock
        FwdDeviceGemmTraits::kMaskingSpec,                      // MaskingSpecialization
        FwdDeviceGemmTraits::kIsDeterministic>;
// type alias for DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle with head_dim = 128
template <typename FwdDeviceGemmTraits>
using DeviceGemmHeadDim128 = device_op::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle<
        FwdDeviceGemmTraits::kNumDimG, 
        FwdDeviceGemmTraits::kNumDimM, 
        FwdDeviceGemmTraits::kNumDimN, 
        FwdDeviceGemmTraits::kNumDimK, 
        FwdDeviceGemmTraits::kNumDimO, 
        typename FwdDeviceGemmTraits::ADataType,
        typename FwdDeviceGemmTraits::B0DataType,
        typename FwdDeviceGemmTraits::B1DataType,
        typename FwdDeviceGemmTraits::CDataType,
        typename FwdDeviceGemmTraits::GemmDataType,
        typename FwdDeviceGemmTraits::ZDataType,
        typename FwdDeviceGemmTraits::LSEDataType,
        typename FwdDeviceGemmTraits::Acc0BiasDataType,
        typename FwdDeviceGemmTraits::Acc1BiasDataType,
        typename FwdDeviceGemmTraits::AccDataType,
        typename FwdDeviceGemmTraits::CShuffleDataType,
        typename FwdDeviceGemmTraits::AElementOp,
        typename FwdDeviceGemmTraits::B0ElementOp,
        typename FwdDeviceGemmTraits::Acc0ElementOp,
        typename FwdDeviceGemmTraits::B1ElementOp,
        typename FwdDeviceGemmTraits::CElementOp,
        FwdDeviceGemmTraits::kGemmSpec,
        FwdDeviceGemmTraits::kTensorSpecA,
        FwdDeviceGemmTraits::kTensorSpecB0,
        FwdDeviceGemmTraits::kTensorSpecB1,
        FwdDeviceGemmTraits::kTensorSpecC,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,         // KPerBlock
        128,    // Gemm1NPerBlock
        32,    // Gemm1KPerBlock
        8,                 // AK1
        8,                 // BK1
        2,                 // B1K1
        32,           // MPerXDL
        32,           // NPerXDL
        1,                 // MXdlPerWave
        4,       // NXdlPerWave
        4,  // Gemm1NXdlPerWave
        device_gemm_trait::S<4, 64, 1>,    // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,   // ABlockLdsExtraM
        device_gemm_trait::S<4, 64, 1>,    // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>,
        device_gemm_trait::S<1, 0, 2>,
        2,
        8,
        8,
        true,  // B0BlockLdsExtraN
        device_gemm_trait::S<8, 32, 1>,   // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>,
        device_gemm_trait::S<0, 2, 1>,
        1,
        4,    //B1BlockTransferSrcScalarPerVector
        2,
        false,
        1,                                    // CShuffleMXdlPerWavePerShuffle
        2,        // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 32, 1, 8>,  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8,                                    // CShuffleBlockTransferScalarPerVector_NPerBlock
        FwdDeviceGemmTraits::kMaskingSpec,                      // MaskingSpecialization
        FwdDeviceGemmTraits::kIsDeterministic>;                       // MaskingSpecialization
} // namespace fwd_device_gemm