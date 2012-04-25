#ifndef _PRIMAL_DUAL_UPDATE_
#define _PRIMAL_DUAL_UPDATE_


#include <cuda_runtime.h>
#include <vector_types.h>
#include <TooN/TooN.h>
#include "cumath.h"
#include "derivatives.h"

void updatePrimalu(float* d_u,
                   float* d_px, float* d_py,
                   float* d_dataterm,
                   const float tau,
                   int2 d_imageSize,
                   const int stridef1 );

void updateDualq( float* d_q0, float* d_q1,
                  float* d_q2, float* d_q3,
                  float* d_v0, float* d_v1,
                  float sigma,
                  int2 d_imageSize,
                  const unsigned int stridef1,
                  const float alpha1);

void updatePrimalv(float* d_q0, float* d_q1,
                   float* d_q2, float* d_q3,
                   float* d_v0, float* d_v1,
                   float* d_px, float* d_py,
                   float tau,
                   int2 d_imageSize,
                   unsigned int stridef1,
                   const float alpha1);

void updateDualp(float* d_px, float* d_py,
                 float* d_u, float* d_v0,
                 float* d_v1, float sigma,
                 float alpha0, int2 d_imageSize,
                 const unsigned int stridef1);




void doOneIterationUpdateDualReg (float* d_px,
                                  float* d_py,
                                  float* d_u,
                                  const unsigned int stride,
                                  const unsigned int width,
                                  const unsigned int height,
                                  const float lambda,
                                  const float sigma_u,
                                  const float sigma_q,
                                  const float sigma_p, const float epsilon);

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
                                const float4 *d_data_term,
                                float2 *critical_points,
                                float2 *grad_vals,
//                                const float* d_data_term,
//                                const float* d_gradient_term,
                                const float *d_px,
                                const float *d_py,
                                const float* d_q,
                                const unsigned int width,
                                const unsigned int height,
                                const unsigned int stridef4,
                                const unsigned int stridef2,
                                const unsigned int stridef1,
                                const float lambda, const float sigma_u, const float sigma_q, const float sigma_p);

void doComputeImageGradient_wrt_depth(const float2 fl,
                                    const float2 pp,
                                    float* d_u0,
                                    float4 *d_data_term,
                                    float2 *critical_points,
                                    float2 *gradients_images,
                                    TooN::Matrix<3,3>& R_lr_,
                                    TooN::Matrix<3,1>& t_lr_,
                                    float *d_ref_img,
                                    float* d_cur_img,
                                    const unsigned int width,
                                    const unsigned int height,
                                    const unsigned int stridef4,
                                    const unsigned int stridef2,
                                    const unsigned int stridef1,
                                    bool disparity, float dmin, float dmax );


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

void doSortCriticalPoints(float2* critical_points,
                          float2* gradients_images,
                          const unsigned int width,
                          const unsigned int height,
                          const unsigned int stridef2);

#endif
