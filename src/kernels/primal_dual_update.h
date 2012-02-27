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
                                  const float sigma_p);

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

void  doOneIterationUpdatePrimal ( float* d_u,
                                  const float* d_u0,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float* d_data_term,
                                 const float* d_gradient_term,
                                 const float* d_px,
                                 const float* d_py,
                                 const float* d_q,
                                 const float lambda,
                                 const float sigma_u,
                                 const float sigma_q,
                                 const float sigma_p,
                                 const int _nimages,
                                 const int slice_stride,
                                 unsigned char* d_sortedindices);

void doComputeImageGradient_wrt_depth(const float2 fl,
                                    const float2 pp,
                                    float* d_u,
                                    float* d_u0,
                                    float* d_data_term,
                                    float* d_gradient_term,
                                    TooN::Matrix<3,3>R_lr_,
                                    TooN::Matrix<3,1>t_lr_,
                                    const unsigned int stride,
                                    float* d_ref_img,
                                    const unsigned int width,
                                    const unsigned int height,
                                    bool disparity,
                                    float dmin,
                                    float dmax,
                                    const unsigned int which_image,
                                    const unsigned int slice_stride);



//doDatatermSorting( d_data_term->data(),
//                   d_gradient_term->data(),
//                   d_data_term->slice_stride(),
//                   d_sortedindices->stride(),
//                   d_u0->data(),
//                   d_u0->stride(),
//                   d_sortedindices->data(),
//                   d_sortedindices->slice_stride(),
//                   d_sortedindices->stride()
//                  );
void doDatatermSorting(float *d_data_term,
                       float *d_gradient_term,
                       unsigned int data_slice_stride,
                       unsigned int d_data_stride,
                       float *d_u0,
                       unsigned int d_u0_stride,
                       unsigned char *d_sortedindices,
                       unsigned int d_sortedindices_slice_stride,
                       unsigned int d_sortedindices_stride,
                       unsigned int width,
                       unsigned int height,
                       const int _nimages
                       );

//doDatatermSorting( d_data_term->data(),
//                   d_gradient_term->data(),
//                   d_data_term->slice_stride(),
//                   d_data_term->stride(),
//                   d_u0->data(),
//                   d_u0->stride(),
//                   d_sortedindices->data(),
//                   d_sortedindices->slice_stride(),
//                   d_sortedindices->stride(),
//                   _nimages
//                  );

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

void BindDataImageStack ( const cudaArray *d_volume,
                          const unsigned int width,
                          const unsigned int height,
                          const unsigned int depth,
                          cudaChannelFormatDesc channelDesc);

void obtainImageSlice(const int which_image, float *d_dest_img, const unsigned int stride, const unsigned int width, const unsigned int height);



#endif
