#include <cuda_runtime.h>
#include <vector_types.h>
#include <TooN/TooN.h>
#include "cumath.h"

void launch_depth_kernel(float* dI2, unsigned int width, unsigned int height, unsigned int imgStride);


//void launch_depth_estimation_kernel_p(float* dpx, float *dpy, float *du,
//                                          unsigned int width, unsigned int height, unsigned int stride,
//                                          float sigma_p, float dmin, float dmax);

void launch_depth_estimation_kernel_p(float* dpx, float *dpy, float *du,
                                          unsigned int width, unsigned int height, unsigned int stride,
                                       float lambda,
                                          float sigma_p, float dmin, float dmax);

void launch_depth_estimation_kernel_q (float* dq, float *du, float *du0,  float sigma_q, float lambda, float *dI1,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int stride,
                                       float *data_term);

void launch_depth_estimation_kernel_u(float* dpx, float* dpy,float *du, float *du0,float *dq, float *dI1,
                               float sigma_u, float lambda, unsigned int width,
                               unsigned int height, unsigned int stride,
                                      float dmin, float dmax, float *grad_wrt_d_at_d0);

void launch_depth_estimation_kernel_I2warped(float* dI2warped,float* du,unsigned int width,
                                      unsigned int height,unsigned int stride,
                                             const float2 fl,
                                             const float2 pp,
                                             TooN::Matrix<3,3> R_lr_,
                                             TooN::Matrix<3,1> t_lr_,
//                                             const TooN::Matrix<3,3>& K,
//                                             const TooN::Matrix<3,3>& Kinv,
//                                             const TooN::Matrix<3,3>& R_lr,
//                                             const TooN::Matrix<3,1>& Kt,
                                             float dmin,
                                             float dmax);

//void launch_depth_estimation_kernel_compute_data_term_and_gradient(const float *dI1, float *data_term,
//                                                                   float *grad_wrt_d_at_d0,
//                                                                   float *du,
//                                                                   float *du0,
//                                                                   const unsigned int height,
//                                                                   const unsigned int width,
//                                                                   const unsigned int stride,
//                                                                   const TooN::Matrix<3,3>& K,
//                                                                   const TooN::Matrix<3,3>& Kinv,
//                                                                   const TooN::Matrix<3,3>& R_lr,
//                                                                   const TooN::Matrix<3,1>& Kt,
//                                                                   float dmin,
//                                                                   float dmax);


void launch_depth_estimation_kernel_compute_data_term_and_gradient(const float *dI1, float *data_term,
                                                                   float *grad_wrt_d_at_d0,
                                                                   float *du,
                                                                   float *du0,
                                                                   const unsigned int height,
                                                                   const unsigned int width,
                                                                   const unsigned int stride,
                                                                   const float2 fl,
                                                                   const float2 pp,
                                                                   TooN::Matrix<3,3>R_lr_,
                                                                   TooN::Matrix<3,1>t_lr_,
//                                                                   const cumat<3,1>t
/*                                                                   const TooN::Matrix<3,3>& K,
                                                                   const TooN::Matrix<3,3>& Kinv,
                                                                   const TooN::Matrix<3,3>& R_lr,
                                                                   const TooN::Matrix<3,1>& Kt,*/
                                                                   float dmin,
                                                                   float dmax);

void launch_kernel_check_KRt_is_correct(float *du,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int stride,
                                        const TooN::Matrix<3,3>& K,
                                        const TooN::Matrix<3,3>& Kinv,
                                        const TooN::Matrix<3,3>& R_lr,
                                        const TooN::Matrix<3,1>& Kt);


void launch_kernel_check_grad_wrt_d0_is_correct(float *du0,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int stride,
                                        const TooN::Matrix<3,3>& K,
                                        const TooN::Matrix<3,3>& Kinv,
                                        const TooN::Matrix<3,3>& R_lr,
                                        const TooN::Matrix<3,1>& Kt);

void  launch_depth_estimation_kernel_copy_u0_to_u(float *du,float *du0, unsigned int width, unsigned int height, unsigned int stride);
