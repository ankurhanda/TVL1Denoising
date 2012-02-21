#ifndef _PRIMAL_DUAL_UPDATE_
#define _PRIMAL_DUAL_UPDATE_


#include <cuda_runtime.h>
#include <vector_types.h>
#include <TooN/TooN.h>
#include "cumath.h"

void doOneIterationUpdateDualReg(float2* d_dual_reg,
                                 float* d_primal,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float lambda,
                                 const float sigma_primal,
                                 const float sigma_dual_data,
                                 const float sigma_dual_reg);

void doOneIterationUpdateDualData(float* d_dual_data,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float* d_data_term,
                                 const float lambda,
                                 const float sigma_primal,
                                 const float sigma_dual_data,
                                 const float sigma_dual_reg);

void  doOneIterationUpdatePrimal ( float* d_primal,
                                   const unsigned int stride,
                                   const unsigned int width,
                                   const unsigned int height,
                                   const float* d_gradient_term,
                                   const float2* d_dual_reg,
                                   const float* d_dual_data,
                                   const float lambda,
                                   const float sigma_primal,
                                   const float sigma_dual_data,
                                   const float sigma_dual_reg);

void doComputeImageGradient_wrt_depth(const float2 fl,
                                   const float2 pp,
                                   float* d_primal,
                                   float* d_primal_u0,
                                   float* d_data_term,
                                   float* d_gradient_term,
                                   TooN::Matrix<3,3> R_lr_,
                                   TooN::Matrix<3,1> t_lr_,
                                   const unsigned stride,
                                   float* d_ref_img,
                                   const unsigned int width,
                                   const unsigned int height);

void doImageWarping(const float2 fl,
                    const float2 pp,
                    TooN::Matrix<3,3> R_lr_,
                    TooN::Matrix<3,1> t_lr_,
                    float *d_cur2ref_warped,
                    float *d_primal,
                    const unsigned int stride,
                    const unsigned int width,
                    const unsigned int height);

void BindDepthTexture        (float* cur_img,
                              unsigned int width,
                              unsigned int height,
                              unsigned int imgStride);

#endif
