#ifndef _PRIMAL_DUAL_UPDATE_
#define _PRIMAL_DUAL_UPDATE_


#include <cuda_runtime.h>
#include <vector_types.h>
#include <TooN/TooN.h>
#include "cumath.h"

void doOneIterationUpdateDualReg (float* d_px,
                                  float* d_py,
                                  float* d_u,
                                  const unsigned int stride,
                                  const unsigned int width,
                                  const unsigned int height,
                                  const float lambda,
                                  const float sigma_u,
                                  const float sigma_q,
                                  const float sigma_p, const float epsilon, const bool use_diffusion_tensor);

void doOneIterationUpdateDualData( float* d_q,
                                  const unsigned int stride,
                                  const unsigned int width,
                                  const unsigned int height,
                                  const float* d_data_term,
                                  const float* d_gradient_term,
                                  float* d_u,
                                  float* d_u0,
                                  const float lambda,
                                  const float sigma_u,
                                  const float sigma_q,
                                  const float sigma_p);

void doOneIterationUpdatePrimal ( float* d_u,
                                const float* d_u0,
                                const unsigned int stride,
                                const unsigned int width,
                                const unsigned int height,
//                                const float* d_data_term,
//                                const float* d_gradient_term,
                                const float4* d_data_term,
                                const unsigned int data_stride,
                                const float* d_px,
                                const float* d_py,
                                const float* d_q,
                                const float lambda,
                                const float sigma_u,
                                const float sigma_q,
                                const float sigma_p, const bool use_diffusion_tensor);

void doComputeImageGradient_wrt_depth(const float2 fl,
                                    const float2 pp,
                                    float* d_u,
                                    float* d_u0,
                                    float4* d_data_term,
                                    const unsigned int data_stride,
//                                    float* d_data_term,
//                                    float* d_gradient_term,
                                    TooN::Matrix<3,3>R_lr_,
                                    TooN::Matrix<3,1>t_lr,
                                    const unsigned int stride,
                                    float* d_ref_img,
                                    const unsigned int width,
                                    const unsigned int height,
                                    bool disparity,
                                    float dmin,
                                    float dmax );

void buildDiffusionTensor(float* d_ref_image,
                          float* d_u0,
                          const unsigned int ref_stride,
                          float4* d_diffusion_tensor,
                          const unsigned int tensor_stride, const unsigned int width, const unsigned int height, const float alpha, const float beta);


void doImageWarping(const float2 fl,
                    const float2 pp,
                    TooN::Matrix<3,3> R_lr_,
                    TooN::Matrix<3,1> t_lr_,
                    float *d_cur2ref_warped,
                    float *d_u,
                    const unsigned int stride,
                    const unsigned int width,
                    const unsigned int height,
                    bool disparity);

void BindDepthTexture        (float* cur_img,
                              unsigned int width,
                              unsigned int height,
                              unsigned int imgStride);

void BindDiffusionTensor(float4* d_diffusion_tensor,
                      unsigned int width,
                      unsigned int height,
                      unsigned int tensor_pitch);

#endif
