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

#include "backward_device_gemm_instance_generator.h"

#include "device_gemm_trait.h"

namespace device_gemm_instance_generator {
// constructor
template<device_gemm_trait::Backward kBackwardDeviceGemmTraits>
BackwardDeviceGemmInstanceGenerator<kBackwardDeviceGemmTraits>::BackwardDeviceGemmInstanceGenerator() {
  // empty
} // end of constructor

auto BackwardDeviceGemmInstanceGenerator::get_instance_head_dim_32(const bool &is_deterministic, const bool &is_casual) {
  
  template <bool kIsDeterministic, device_gemm_trait::MaskingSpec kMaskingSpec>
  using InstanceOfHeadDim32 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          device_gemm_trait::kNumDimG, 
          device_gemm_trait::kNumDimM, 
          device_gemm_trait::kNumDimN, 
          device_gemm_trait::kNumDimK, 
          device_gemm_trait::kNumDimO, 
          InputDataType, 
          OutputDataType, 
          GemmDataType,
          DropoutType, 
          device_gemm_trait::LSEDataType, 
          device_gemm_trait::Acc0BiasDataType, 
          device_gemm_trait::Acc1BiasDataType,
          device_gemm_trait::AccDataType, 
          device_gemm_trait::ShuffleDataType, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::Scale,
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::YElementOp, 
          device_gemm_trait::kGemmSpec, 
          device_gemm_trait::kTensorSpecQ, 
          device_gemm_trait::kTensorSpecK,
          device_gemm_trait::kTensorSpecV, 
          device_gemm_trait::kTensorSpecY, 
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
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_spec, // MaskingSpec
          is_deterministic>; 
} // end of get_instance_head_dim_32
auto BackwardDeviceGemmInstanceGenerator::get_instance_head_dim_64(const bool &is_deterministic, const bool &is_casual) {
  using InstanceOfHeadDim64 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V1<
          device_gemm_trait::kNumDimG, 
          device_gemm_trait::kNumDimM, 
          device_gemm_trait::kNumDimN, 
          device_gemm_trait::kNumDimK, 
          device_gemm_trait::kNumDimO, 
          InputDataType, 
          OutputDataType, 
          GemmDataType,
          DropoutType, 
          device_gemm_trait::LSEDataType, 
          device_gemm_trait::Acc0BiasDataType, 
          device_gemm_trait::Acc1BiasDataType,
          device_gemm_trait::AccDataType, 
          device_gemm_trait::ShuffleDataType, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::Scale,
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::YElementOp, 
          device_gemm_trait::kGemmSpec, 
          device_gemm_trait::kTensorSpecQ, 
          device_gemm_trait::kTensorSpecK,
          device_gemm_trait::kTensorSpecV, 
          device_gemm_trait::kTensorSpecY,
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
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_spec, // MaskingSpec
          is_deterministic>; 
} // end of get_instance_head_dim_64
auto BackwardDeviceGemmInstanceGenerator::get_instance_head_dim_128(const bool &is_deterministic, const bool &is_casual) {
  using InstanceOfHeadDim128 = device_op::DeviceGroupedMultiheadAttentionBackward_Xdl_CShuffle_V2<
          device_gemm_trait::kNumDimG, 
          device_gemm_trait::kNumDimM, 
          device_gemm_trait::kNumDimN, 
          device_gemm_trait::kNumDimK, 
          device_gemm_trait::kNumDimO, 
          InputDataType, 
          OutputDataType, 
          GemmDataType,
          DropoutType, 
          device_gemm_trait::LSEDataType, 
          device_gemm_trait::Acc0BiasDataType, 
          device_gemm_trait::Acc1BiasDataType,
          device_gemm_trait::AccDataType, 
          device_gemm_trait::ShuffleDataType, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::Scale,
          device_gemm_trait::QkvElementOp, 
          device_gemm_trait::YElementOp, 
          device_gemm_trait::kGemmSpec, 
          device_gemm_trait::kTensorSpecQ, 
          device_gemm_trait::kTensorSpecK,
          device_gemm_trait::kTensorSpecV, 
          device_gemm_trait::kTensorSpecY,
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
          c_shuffle_block_transfer_scalar_per_vector_n_per_block, // c_shuffle_block_transfer_scalar_per_vector_n_per_block
          masking_spec, // MaskingSpec
          is_deterministic>;
} // end of get_instance_head_dim_128
} // namespace device_gemm_instance_generator