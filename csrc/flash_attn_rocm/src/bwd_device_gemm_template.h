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

namespace bwd_device_gemm {
namespace device_op = ck::tensor_operation::device; // namespace alias for internal use
// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1 of head dim = 32
template <typename BwdDeviceGemmTraits>
using DeviceGemmHeadDim32 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
        BwdDeviceGemmTraits::kNumDimG, 
        BwdDeviceGemmTraits::kNumDimM, 
        BwdDeviceGemmTraits::kNumDimN, 
        BwdDeviceGemmTraits::kNumDimK,
        BwdDeviceGemmTraits::kNumDimO, 
        BwdDeviceGemmTraits::InputDataType, 
        BwdDeviceGemmTraits::OutputDataType, 
        BwdDeviceGemmTraits::GemmDataType,
        BwdDeviceGemmTraits::ZDataType, 
        BwdDeviceGemmTraits::LSEDataType, 
        BwdDeviceGemmTraits::Acc0BiasDataType, 
        BwdDeviceGemmTraits::Acc1BiasDataType,
        BwdDeviceGemmTraits::AccDataType, 
        BwdDeviceGemmTraits::ShuffleDataType, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::Scale,
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::YElementOp, 
        BwdDeviceGemmTraits::kGemmSpec, 
        BwdDeviceGemmTraits::kTensorSpecQ, 
        BwdDeviceGemmTraits::kTensorSpecK,
        BwdDeviceGemmTraits::kTensorSpecV, 
        BwdDeviceGemmTraits::kTensorSpecY, 
        1, 
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        32,          // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        1,           // Gemm1NXdlPerWave
        1,           // Gemm2NXdlPerWave
        device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<8, 32, 1>, // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>, 
        device_gemm_trait::S<0, 2, 1>, 
        1, 
        4, 
        2, 
        false,
        1, // CShuffleMXdlPerWavePerShuffle
        1, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 64, 1, 4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        BwdDeviceGemmTraits::kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        BwdDeviceGemmTraits::kMaskingSpec, // MaskingSpec
        BwdDeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1 of head dim = 64
template <typename BwdDeviceGemmTraits>
using DeviceGemmHeadDim64 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
        BwdDeviceGemmTraits::kNumDimG, 
        BwdDeviceGemmTraits::kNumDimM, 
        BwdDeviceGemmTraits::kNumDimN, 
        BwdDeviceGemmTraits::kNumDimK, 
        BwdDeviceGemmTraits::kNumDimO, 
        BwdDeviceGemmTraits::InputDataType, 
        BwdDeviceGemmTraits::OutputDataType, 
        BwdDeviceGemmTraits::GemmDataType,
        BwdDeviceGemmTraits::ZDataType, 
        BwdDeviceGemmTraits::LSEDataType, 
        BwdDeviceGemmTraits::Acc0BiasDataType, 
        BwdDeviceGemmTraits::Acc1BiasDataType,
        BwdDeviceGemmTraits::AccDataType, 
        BwdDeviceGemmTraits::ShuffleDataType, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::Scale,
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::YElementOp, 
        BwdDeviceGemmTraits::kGemmSpec, 
        BwdDeviceGemmTraits::kTensorSpecQ, 
        BwdDeviceGemmTraits::kTensorSpecK,
        BwdDeviceGemmTraits::kTensorSpecV, 
        BwdDeviceGemmTraits::kTensorSpecY, 
        1, 
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        64,          // KPerBlock
        64,          // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        2,           // Gemm1NXdlPerWave
        2,           // Gemm2NXdlPerWave
        device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<8, 32, 1>, // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>, 
        device_gemm_trait::S<0, 2, 1>, 
        1, 
        4, 
        2, 
        false,
        1, // CShuffleMXdlPerWavePerShuffle
        2, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        BwdDeviceGemmTraits::kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        BwdDeviceGemmTraits::kMaskingSpec, // MaskingSpec
        BwdDeviceGemmTraits::kIsDeterministic>;

// type alias for DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2 of head dim = 128
template <typename BwdDeviceGemmTraits>
using DeviceGemmHeadDim128 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
        BwdDeviceGemmTraits::kNumDimG, 
        BwdDeviceGemmTraits::kNumDimM, 
        BwdDeviceGemmTraits::kNumDimN, 
        BwdDeviceGemmTraits::kNumDimK, 
        BwdDeviceGemmTraits::kNumDimO, 
        BwdDeviceGemmTraits::InputDataType, 
        BwdDeviceGemmTraits::OutputDataType, 
        BwdDeviceGemmTraits::GemmDataType,
        BwdDeviceGemmTraits::ZDataType, 
        BwdDeviceGemmTraits::LSEDataType, 
        BwdDeviceGemmTraits::Acc0BiasDataType, 
        BwdDeviceGemmTraits::Acc1BiasDataType,
        BwdDeviceGemmTraits::AccDataType, 
        BwdDeviceGemmTraits::ShuffleDataType, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::Scale,
        BwdDeviceGemmTraits::QkvElementOp, 
        BwdDeviceGemmTraits::YElementOp, 
        BwdDeviceGemmTraits::kGemmSpec, 
        BwdDeviceGemmTraits::kTensorSpecQ, 
        BwdDeviceGemmTraits::kTensorSpecK,
        BwdDeviceGemmTraits::kTensorSpecV, 
        BwdDeviceGemmTraits::kTensorSpecY,
        1, 
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        64,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        4,           // Gemm1NXdlPerWave
        2,           // Gemm2NXdlPerWave
        device_gemm_trait::S<4, 64, 1>, // ABlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<4, 64, 1>, // BBlockTransfer
        device_gemm_trait::S<1, 0, 2>, 
        device_gemm_trait::S<1, 0, 2>, 
        2, 
        8, 
        8, 
        true,
        device_gemm_trait::S<8, 32, 1>, // B1BlockTransfer
        device_gemm_trait::S<0, 2, 1>, 
        device_gemm_trait::S<0, 2, 1>, 
        1, 
        4, 
        2, 
        false,
        1, // CShuffleMXdlPerWavePerShuffle
        4, // CShuffleNXdlPerWavePerShuffle
        device_gemm_trait::S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        BwdDeviceGemmTraits::kCShuffleBlockTransferScalarPerVectorNPerBlock, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
        BwdDeviceGemmTraits::kMaskingSpec, // MaskingSpec
        BwdDeviceGemmTraits::kIsDeterministic>;
} // namespace bwd_device_gemm