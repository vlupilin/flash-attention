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

#include "fmha_fprop_fp16_bf16_kernel.gfx90a.h"

#include <vector>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "fmha.h"
#include "fp16_switch.h"


template <typename InputType, 
          typename OutputType,
          typename GemmDataType,
          typename DropoutType,
          ck::index_t version, 
          ck::index_t c_shuffle_block_transfer_scalar_per_vector_n_per_block,
          MaskingSpecialization masking_specialization>
void run_fmha_dgrad_fp16_bf16_gfx90a_loop_(LaunchParams<FmhaDgradParams> &launch_params) {
  
  
  // init the instance with parameters
  auto run_kernel = [&]<typename DeviceGemmInstance>(DeviceGemmInstance gemm) {
    
  };

    if (version == 1) {
      
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else if (version == 2) {
      
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    } else {
      
      auto gemm = DeviceGemmInstance{};
      run_kernel(gemm);
    }
  
  }
}

void run_fmha_dgrad_fp16_bf16_gfx90a(LaunchParams<FmhaDgradParams> &launch_params) {
  using BFloat16 = ck::bhalf_t;

  if (launch_params.params.is_performance_mode) {
    FP16_SWITCH(launch_params.params.is_bf16, [&] {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 1, 4, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 2, 4, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 3, 4, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 1, 4, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 2, 4, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, BFloat16, DropOutType, 3, 4, kMaskingSpecializationDefault>(launch_params);
        }
      }
    }); 
  // non-performance mode
  } else {
    FP16_SWITCH(launch_params.params.is_bf16, [&] {
      if (launch_params.params.is_causal) {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 1, 4, kMaskingSpecializationCausal>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 2, 4, kMaskingSpecializationCausal>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 3, 4, kMaskingSpecializationCausal>(launch_params);
        }
      } else {
        if (launch_params.params.d > 64) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 1, 4, kMaskingSpecializationDefault>(launch_params);
        } else if (launch_params.params.d > 32) {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 2, 4, kMaskingSpecializationDefault>(launch_params);
        } else {
          run_fmha_dgrad_fp16_bf16_gfx90a_loop_<DataType, DataType, DataType, DropOutType, 3, 4, kMaskingSpecializationDefault>(launch_params);
        }
      }
    }); 
  }
}