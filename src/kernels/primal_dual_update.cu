#define HAVE_TOON
#undef isfinite
#undef isnan

#include <math.h>
#include <TooN/TooN.h>
#include <TooN/Cholesky.h>
#include <TooN/LU.h>
#include <boost/math/common_factor.hpp>

#include <stdio.h>
#include <cutil_inline.h>
#include "cumath.h"
#include "primal_dual_update.h"
#include <iostream>



texture<float, 2, cudaReadModeElementType> TexImgCur;
const static cudaChannelFormatDesc chandesc_float1 =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

texture<float4,2,cudaReadModeElementType> DiffusionTensor;
const static cudaChannelFormatDesc chandesc_float4 =
cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);


__global__ void kernel_doOneIterationUpdatePrimal (float * d_u,
                                                   const float* d_u0,
                                                   const unsigned int stride,
                                                   const unsigned int width,
                                                   const unsigned int height,
                                                   const float4* d_data_term,
                                                   const unsigned int data_stride,
//                                                   const float* d_data_term,
//                                                   const float* d_gradient_term,
                                                   const float* d_px,
                                                   const float* d_py,
                                                   const float* d_q,
                                                   const float lambda,
                                                   const float sigma_u,
                                                   const float sigma_q,
                                                   const float sigma_p,
                                                   const bool use_diffusion_tensor)
{


    /// Update Equations should be
    /// u = u - tau*( lambda*q*grad - divp )

    float dxp = 0 , dyp = 0, dxpy = 0, dypx = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x >= 1 && x < width  )
    {
        dxp  = d_px[y*stride+x] - d_px[y*stride+(x-1)];

//       dxpy  = d_py[y*stride+x] - d_py[y*stride+(x-1)];
   }

    if ( y >= 1 && y < height )
    {
        dyp  = d_py[y*stride+x] - d_py[(y-1)*stride+x];

//        dypx = d_px[y*stride+x] - d_px[(y-1)*stride+x];
    }

    float div_p = dxp + dyp;

//    float u_ = d_u[y*stride+x] + sigma_u*(div_p);

    /// case 1
    /// when rho_1(u_) > sigma_u*(a1-a2)*a1 &&
    /// rho_2(u_) < sigma_u*(a1-a2)*a2
    /// u^{n+1} = u_ - sigma_u*(a1-a2)

    /// case 2
    /// when rho_2(u_) > sigma_u*(a1+a2)*a2
    /// u^{n+1} = u_ - sigma_u*(a1+a2)

    /// case 3
    /// when rho_1(u_) < -sigma_u*(a1+a2)*a1
    /// u^{n+1} = u_ + sigma_u*(a1+a2)

    /// case 4
    /// It is one of the critical points
    /// u^{n+1} = min(E(d)V{d \in \frac{bi}{ai}}



//    float ref_img_val = d_u0[y*stride+x];
//    float u_update = d_u[y*stride+x] + sigma_u*div_p + lambda*sigma_u*ref_img_val;

//    d_u[y*stride+x] = u_update/(1+lambda*sigma_u);


//    if ( use_diffusion_tensor )
//    {
//        float4 tensor_element = tex2D(DiffusionTensor,x+0.5,y+0.5);

//        float a11 = tensor_element.x;
//        float a12 = tensor_element.y;
//        float a21 = tensor_element.z;
//        float a22 = tensor_element.w;

//        div_p = a11*dxp + a12*dxpy + a21*dypx + a22*dyp;
//        //div_p = a11*dxp +  a11*dyp;
//    }


//    float u_update = d_u[y*stride+x] + sigma_u*div_p - sigma_u*lambda*d_q[y*stride+x]*d_gradient_term[y*stride+x];
//    d_u[y*stride+x] = u_update;
//    d_u[y*stride+x] = fmaxf(0.0f,fminf(1.0f,u_update));


//    float grad_sqr = 1;//d_gradient_term[y*stride+x]*d_gradient_term[y*stride+x];

//    float u_ = (d_u[y*stride+x] + sigma_u*(div_p));

////    float u0 = d_u0[y*stride+x];

////    float rho = d_data_term[y*stride+x] + (u_-u0)*d_gradient_term[y*stride+x];

//    float rho = u_ - ref_img_val;

//    if ( rho < -sigma_u*lambda*grad_sqr)

//        d_u[y*stride+x] =  u_ + sigma_u*lambda*1;//d_gradient_term[y*stride+x];

//    else if( rho > sigma_u*lambda*grad_sqr)

//        d_u[y*stride+x] =  u_ - sigma_u*lambda*1;//d_gradient_term[y*stride+x];

//    else if ( fabs(rho) <= sigma_u*lambda*grad_sqr)
//        d_u[y*stride+x] =  u_ - rho;



//    float grad = 0;//d_data_term[y*data_stride+x].z;
//    float a_b  = 0;//(d_data_term[y*data_stride+x].x - d_data_term[y*data_stride+x].y);

//    int patch_hf_width = 0;

//    int count = 0;

//    float mu_cur=0,mu_ref=0,mu_grad=0;
//    float sum_I2_times_I2grad=0, sum_I2grad=0, sum_I2=0;
//    float sum_I1_times_I2grad=0, sum_I2_sqr=0, sum_I1=0;


//    for(int i = -patch_hf_width ; i <= patch_hf_width+1 ; i++ )
//    {
//        for(int j = -patch_hf_width ; j <= patch_hf_width+1 ; j++ )
//        {
//            if ( x+i >= 0 && x+i< width && y+j >=0 && y+j < height)
//            {
//                float4 data_vars = d_data_term[(y+j)*data_stride+(x+i)];

//                mu_cur  += data_vars.x;
//                mu_ref  += data_vars.y;
//                mu_grad += data_vars.z;

////                sum_I2_times_I2grad += data_vars.x*data_vars.z;
////                sum_I2grad += data_vars.z;
////                sum_I2  += data_vars.x;

////                sum_I2_sqr = data_vars.z*data_vars.z;

////                sum_I1_times_I2grad += data_vars.y*data_vars.z;
////                sum_I1 += data_vars.y;

//                count++;
//            }
//        }
//    }

//    mu_cur  = mu_cur/(float)count;
//    mu_ref  = mu_ref/(float)count;
//    mu_grad = mu_grad/(float)count;

//    float sum_a_b_grad = 0;
//    float grad_sqr=0;

//    sum_a_b_grad = sum_I2_times_I2grad - mu_cur*sum_I2grad - mu_grad*sum_I2 + count*mu_cur*mu_grad;
//    sum_a_b_grad -= (sum_I1_times_I2grad - mu_ref*sum_I2grad - mu_grad*sum_I1 + count*mu_ref*mu_grad);
//    grad_sqr = sum_I2_sqr + count*mu_grad*mu_grad - 2*mu_grad*sum_I2grad;


//    for(int i = -patch_hf_width ; i <= patch_hf_width+1 ; i++ )
//    {
//        for(int j = -patch_hf_width ; j <= patch_hf_width+1 ; j++ )
//        {
//            if ( x+i >= 0 && x+i< width && y+j >=0 && y+j < height)
//            {

//                float grad = d_data_term[(y+j)*data_stride+x+i].z - mu_grad;
//                float a = d_data_term[(y+j)*data_stride+(x+i)].x - mu_cur;
//                float b = d_data_term[(y+j)*data_stride+(x+i)].y - mu_ref;

//                sum_a_b_grad += (a-b)*grad;
//                grad_sqr += grad*grad;

//            }
//        }
//    }


//    sum_a_b_grad = sum_a_b_grad   / (float)count;
//    grad_sqr = grad_sqr  / (float)count;

//    float diff_term = 2*lambda*(sum_a_b_grad + grad_sqr*(d_u[y*stride+x]-d_u0[y*stride+x])) - div_p;
//    float d_u_ = d_u[y*stride+x] - sigma_u*(diff_term);
//    d_u[y*stride+x] = d_u_;

    float grad = d_data_term[y*data_stride+x].z;
    float a_b  = (d_data_term[y*data_stride+x].x - d_data_term[y*data_stride+x].y);

    float grad_sqr = grad*grad;

    float u_ = (d_u[y*stride+x] + sigma_u*(div_p));

    float u0 = d_u0[y*stride+x];

    float rho = a_b + (u_-u0)*grad;

    if ( rho < -sigma_u*lambda*grad_sqr)

        d_u[y*stride+x] =  u_ + sigma_u*lambda*grad;

    else if( rho > sigma_u*lambda*grad_sqr)

        d_u[y*stride+x] =  u_ - sigma_u*lambda*grad;

    else if ( fabs(rho) <= sigma_u*lambda*grad_sqr)
        d_u[y*stride+x] =  u_ - rho/(grad+10E-6);


    d_u[y*stride+x] = fmaxf(0.0f,fminf(1.0f,d_u[y*stride+x]));

}

void  doOneIterationUpdatePrimal ( float* d_u,
                                  const float* d_u0,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float4* d_data_term,
                                 const unsigned int data_stride,
//                                 const float* d_data_term,
//                                 const float* d_gradien_term,
                                 const float* d_px,
                                 const float* d_py,
                                 const float* d_q,
                                 const float lambda,
                                 const float sigma_u,
                                 const float sigma_q,
                                 const float sigma_p,
                                  const bool use_diffusion_tensor)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdatePrimal<<<grid,block>>>(d_u,
                                                      d_u0,
                                                       stride,
                                                       width,
                                                       height,
                                                       d_data_term,
                                                       data_stride,
//                                                       d_gradien_term,
                                                       d_px,                                                      
                                                       d_py,
                                                       d_q,
                                                       lambda,
                                                       sigma_u,
                                                       sigma_q,
                                                       sigma_p,
                                                      use_diffusion_tensor);



}










__global__ void kernel_doOneIterationUpdateDualReg (float* d_px,
                                                    float* d_py,
                                                    float* d_u,
                                                    const unsigned int stride,
                                                    const unsigned int width,
                                                    const unsigned int height,
                                                    const float lambda,
                                                    const float sigma_u,
                                                    const float sigma_q,
                                                    const float sigma_p,
                                                    const float epsilon,
                                                    bool use_diffusion_tensor)
{


    /// Update Equations should be
    /// p = p - sigma_p*( grad_d )
    /// p = p / max(1.0f,length(p))

    float u_dx = 0, u_dy = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x + 1 < width )
    {
        u_dx = d_u[y*stride+(x+1)] - d_u[y*stride+x];
    }

    if ( y + 1 < height )
    {
        u_dy = d_u[(y+1)*stride+x] - d_u[y*stride+x];
    }


//    float a11 = 1, a12 = 0, a21 = 0, a22 = 1;

//    if ( use_diffusion_tensor )
//    {
//        float4 tensor_element = tex2D(DiffusionTensor,x+0.5,y+0.5);

//        a11 = tensor_element.x;
//        a12 = tensor_element.y;
//        a21 = tensor_element.z;
//        a22 = tensor_element.w;
//    }


//    float u_dx_ = u_dx;
//    float u_dy_ = u_dy;

//    u_dx = a11*u_dx_ + a12*u_dy_;
//    u_dy = a21*u_dx_ + a22*u_dy_;

//    u_dx = a11*u_dx_ ;
//    u_dy =  a22*u_dy_;



    float pxval = (d_px[y*stride+x] + sigma_p*(u_dx))/(1+epsilon*sigma_p);
    float pyval = (d_py[y*stride+x] + sigma_p*(u_dy))/(1+epsilon*sigma_p);

    // reprojection
    float reprojection_p   = fmaxf(1.0f,length( make_float2(pxval,pyval) ) );

    d_px[y*stride+x] = pxval/reprojection_p;
    d_py[y*stride+x] = pyval/reprojection_p;


}



void doOneIterationUpdateDualReg (float* d_px,
                                  float* d_py,
                                  float* d_u,
                                  const unsigned int stride,
                                  const unsigned int width,
                                  const unsigned int height,
                                  const float lambda,
                                  const float sigma_u,
                                  const float sigma_q,
                                  const float sigma_p,
                                  const float epsilon,
                                  const bool use_diffusion_tensor)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdateDualReg<<<grid,block>>>(d_px,
                                                       d_py,
                                                       d_u,
                                                       stride,
                                                       width,
                                                       height,
                                                       lambda,
                                                       sigma_u,
                                                       sigma_q,
                                                       sigma_p,
                                                       epsilon,
                                                       use_diffusion_tensor);

}



__global__ void kernel_computeImageGradient_wrt_depth(const float2 fl,
                                               const float2 pp,
                                               float* d_u,
                                               float* d_u0,
                                               float4* d_data_term,
                                               const unsigned int data_stride,
//                                                float* d_data_term,
//                                                float* d_gradient_term,
                                               cumat<3,3>R,
                                               cumat<3,1>t,
                                               const unsigned int stride,
                                               float* d_ref_img,
                                               const unsigned int width,
                                               const unsigned int height,
                                               bool disparity,
                                               float dmin,
                                               float dmax )
{

    if ( disparity)
    {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


        float xinterp0 = (float)x+ d_u0[y*stride+x];
        float xinterp1 = (float)x+ d_u0[y*stride+x]+1;


        float I2_u0      = tex2D(TexImgCur,xinterp0+0.5,(float)y+0.5);
        float I1_val     = d_ref_img[y*stride+x];
        float grad_I2_u0 = tex2D(TexImgCur,xinterp1+0.5,(float)y+0.5) - tex2D(TexImgCur,xinterp0+0.5,(float)y+0.5);

        d_data_term[y*data_stride+x]= make_float4(I2_u0,I1_val,grad_I2_u0,1.0f);
//        d_data_term[y*data_stride+x]= make_float4(0,I1_val,0,1.0f);


//        float u0 = d_u0[y*stride+x];
//        float u  = d_u[y*stride+x];

//        d_gradient_term[y*stride+x] = grad_I2_u0 +10E-6;
//        float data_term_value  = (I2_u0 /*+ (u-u0)*grad_I2_u0 */- I1_val);
//        d_data_term[y*stride+x] = data_term_value;
    }

    else
    {

        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

        /// Checked!
        float2 invfl = 1.0f/fl;

        /// Checked!
        float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);//*(dmax-dmin);
        cumat<3,1> uvnormMat = {uvnorm.x, uvnorm.y, uvnorm.z};


        float zLinearised = d_u0[y*stride+x];
        zLinearised = fmaxf(0.0f,fminf(1.0f,zLinearised));

        cumat<3,1> p3d_r       = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, zLinearised};

        /// Are we really sure of this?
        cumat<3,1> p3d_dest  =  R*p3d_r + t;

        p3d_dest(2,0) = fmax(0.0f,fmin(1.0f,p3d_dest(2,0)));

        float dIdz;
        float Id_minus_Ir;

        float3 p3d_dest_vec = {p3d_dest(0,0), p3d_dest(1,0), p3d_dest(2,0)};

        float2 p2D_live = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

        p2D_live.x= fmaxf(0,fminf(width, p2D_live.x*fl.x + pp.x));
        p2D_live.y= fmaxf(0,fminf(height,p2D_live.y*fl.y + pp.y));


        float Ir =  d_ref_img[y*stride+x];

        float Id =  tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f);
        float Idx = tex2D(TexImgCur,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);

        if ( p2D_live.x+0.5+1 > (float) width)
            Idx = Id;

        float Idy = tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);

        if ( p2D_live.y+0.5+1 > (float) height)
            Idy = Id;


        float2 dIdx = make_float2(Idx-Id, Idy-Id);

        p3d_dest_vec.z = p3d_dest_vec.z + 10E-6;

//        float3 dpi_u = make_float3(1/p3d_dest_vec.z, 0,-(p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
//        float3 dpi_v = make_float3(0, 1/p3d_dest_vec.z,-(p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        float3 dpi_u = make_float3(fl.x/p3d_dest_vec.z, 0,-(fl.x*p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
        float3 dpi_v = make_float3(0, fl.y/p3d_dest_vec.z,-(fl.y*p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        cumat<3,1> dXdz = R*uvnormMat;///(zLinearised*zLinearised);

        float3 dXdz_vec = {dXdz(0,0),dXdz(1,0),dXdz(2,0)};

        dIdz =  dot(dIdx, make_float2( dot(dXdz_vec,dpi_u),  dot(dXdz_vec,dpi_v) ) );

        d_data_term[y*data_stride+x]= make_float4(Id,Ir,dIdz,1);
//        d_data_term[y*stride+x] = Id_minus_Ir ;//+ (u-u0)*dIdz;
//        d_gradient_term[y*stride+x] = dIdz;

    }

}


void doComputeImageGradient_wrt_depth(const float2 fl,
                                    const float2 pp,
                                    float* d_u,
                                    float* d_u0,
                                    float4* d_data_term,
                                    const unsigned int data_stride,
//                                      float* d_data_term,
//                                      float* d_gradient_term,
                                    TooN::Matrix<3,3>R_lr_,
                                    TooN::Matrix<3,1>t_lr_,
                                    const unsigned int stride,
                                    float* d_ref_img,
                                    const unsigned int width,
                                    const unsigned int height,
                                    bool disparity,
                                    float dmin,
                                    float dmax )
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    cumat<3,1> t = cumat_from<3,1,float>(t_lr_);

    kernel_computeImageGradient_wrt_depth<<<grid,block>>>(fl,
                                          pp,
                                          d_u,
                                          d_u0,
                                          d_data_term,
                                          data_stride,
//                                          d_gradient_term,
                                          R,
                                          t,
                                          stride,
                                          d_ref_img,
                                          width,
                                          height,
                                          disparity,
                                          dmin,
                                          dmax);


}


__global__ void kernel_doImageWarping( const float2 fl,
                                       const float2 pp,
                                       const cumat<3,3> R,
                                       const cumat<3,1> t,
                                       float* d_cur2ref_warped,
                                       float* d_u,
                                       const unsigned int stride,
                                       const unsigned int width,
                                       const unsigned int height,
                                       bool disparity)
{


    if ( disparity )
    {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

        float xinterp0 = (float)x+ d_u[y*stride+x];

        d_cur2ref_warped[y*stride+x] = tex2D(TexImgCur,xinterp0+0.5,y);
    }

    else
    {
        unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

        float2 invfl = 1.0f/fl;
        float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);

        float zLinearised = d_u[y*stride+x];

        cumat<3,1> p3d      = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, uvnorm.z*zLinearised};
        cumat<3,1> p3d_dest = R*p3d + t;
        float2 p2D_live     = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

        p2D_live.x = p2D_live.x*fl.x + pp.x;
        p2D_live.y = p2D_live.y*fl.y + pp.y;

        d_cur2ref_warped[y*stride+x] = tex2D(TexImgCur,p2D_live.x,p2D_live.y);
    }
}



void doImageWarping(const float2 fl,
                    const float2 pp,
                    TooN::Matrix<3,3> R_lr_,
                    TooN::Matrix<3,1> t_lr_,
                    float *d_cur2ref_warped,
                    float *d_u,
                    const unsigned int stride,
                    const unsigned int width,
                    const unsigned int height,
                    bool disparity)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    cumat<3,1> t = cumat_from<3,1,float>(t_lr_);

    kernel_doImageWarping<<<grid,block>>>(fl,
                                          pp,
                                          R,
                                          t,
                                          d_cur2ref_warped,
                                          d_u,
                                          stride,
                                          width,
                                          height,
                                          disparity);

}


//__global__ void kernel_doOneIterationUpdateDualData( float* d_q,
//                                             const unsigned int stride,
//                                             const unsigned int width,
//                                             const unsigned int height,
//                                             const float* d_data_term,
//                                             const float* d_gradient_term,
//                                             float* d_u,
//                                             float* d_u0,
//                                             const float lambda,
//                                             const float sigma_u,
//                                             const float sigma_q,
//                                             const float sigma_p)
//{

//    /// Update Equations should be
//    /// q = q - sigma_q*( lambda*data_term )
//    /// q = q / max(1.0f,fabs(q))

//    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

////    float epsilon=0.001;
//    float u = d_u[y*stride+x];
//    float u0 = d_u0[y*stride+x];

//    float q_update = d_q[y*stride+x] + sigma_q*lambda*(d_data_term[y*stride+x]+ (u - u0)*d_gradient_term[y*stride+x]);

////    q_update = q_update/(1+epsilon*sigma_q);

//    float reprojection_q = max(1.0f,fabs(q_update)/*/lambda*/);

//    d_q[y*stride+x] = q_update/reprojection_q;



//}



//void doOneIterationUpdateDualData( float* d_q,
//                                  const unsigned int stride,
//                                  const unsigned int width,
//                                  const unsigned int height,
//                                  const float* d_data_term,
//                                  const float* d_gradient_term,
//                                  float* d_u,
//                                  float* d_u0,
//                                  const float lambda,
//                                  const float sigma_u,
//                                  const float sigma_q,
//                                  const float sigma_p)
//{

//    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
//    dim3 grid( width / block.x, height / block.y);

//    kernel_doOneIterationUpdateDualData<<<grid,block>>>(d_q,
//                                                       stride,
//                                                       width,
//                                                       height,
//                                                       d_data_term,
//                                                       d_gradient_term,
//                                                       d_u,
//                                                       d_u0,
//                                                       lambda,
//                                                       sigma_u,
//                                                       sigma_q,
//                                                       sigma_p);

//}



void BindDepthTexture(float* cur_img,
                      unsigned int width,
                      unsigned int height,
                      unsigned int imgStride)

{
    cudaBindTexture2D(0,TexImgCur,cur_img,chandesc_float1,width, height,imgStride*sizeof(float));

    TexImgCur.addressMode[0] = cudaAddressModeClamp;
    TexImgCur.addressMode[1] = cudaAddressModeClamp;
    TexImgCur.filterMode = cudaFilterModeLinear;
    TexImgCur.normalized = false;    // access with normalized texture coordinates
}


void BindDiffusionTensor(float4* d_diffusion_tensor,
                      unsigned int width,
                      unsigned int height,
                      unsigned int tensor_pitch)

{
    cudaBindTexture2D(0,DiffusionTensor,d_diffusion_tensor,chandesc_float4,width, height,tensor_pitch);

    DiffusionTensor.addressMode[0] = cudaAddressModeClamp;
    DiffusionTensor.addressMode[1] = cudaAddressModeClamp;
    DiffusionTensor.filterMode = cudaFilterModePoint;
    DiffusionTensor.normalized = false;    // access with normalized texture coordinates
}


__global__ void kernel_buildDiffusionTensor(float* d_ref_image,
                                            float* d_u0,
                                            const unsigned int ref_stride,
                                            float4* d_diffusion_tensor,
                                            const unsigned int tensor_stride,
                                            const unsigned int width,
                                            const unsigned int height,
                                            const float alpha,
                                            const float beta)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float Ix = 0, Ix_sqr = 0;
    float Iy = 0, Iy_sqr = 0;

    float xinterp0 = (float)x+d_u0[y*ref_stride+x];

    if ( x + 1 < width )
        Ix =  d_ref_image[y*ref_stride+(x+1)] - d_ref_image[y*ref_stride+x];
    if ( y + 1 < height )
        Iy =  d_ref_image[(y+1)*ref_stride+x] - d_ref_image[y*ref_stride+x];

//    Ix+= tex2D(TexImgCur,xinterp0+1+0.5,y+0.5)  -tex2D(TexImgCur,xinterp0+0.5,y+0.5);
//    Iy+= tex2D(TexImgCur,xinterp0+1+0.5,y+1+0.5)-tex2D(TexImgCur,xinterp0+0.5,y+0.5);
//    Ix/=2;
//    Iy/=2;

    Ix_sqr = (Ix*Ix);
    Iy_sqr = (Iy*Iy);

    float mag_grad = 1.0f;
    float a11 = 1.0f, a12 = 1E-6;//0.0f;
    float a21 = 1E-6, a22 = 1.0f;

//    mag_grad = max(1E-6,Ix_sqr + Iy_sqr);
    mag_grad = (Ix_sqr + Iy_sqr + 1E-6);

    float exp_val = exp(-alpha*pow(mag_grad,beta));
    //float exp_valx = exp(-alpha*pow(sqrtf(Ix_sqr),beta));
    //float exp_valy = exp(-alpha*pow(sqrtf(Iy_sqr),beta));

//    a11  = exp_val;
//    a12  = 0;
//    a21  = 0;
//    a22  = exp_val;
//    mag_grad = 1.0f;


    a22 = Ix_sqr*(exp_val)+Iy_sqr*(1-exp_val);
    a12 = Ix*Iy*(-1.0f+exp_val);
    a21 = a12;
    a11 = Iy_sqr*(exp_val)+Ix_sqr*(1-exp_val);

//    a11 = exp_val;//Ix_sqr*(exp_val);//+Iy_sqr;
//    a12 = 0;//Ix*Iy*(-1.0f+exp_val);
//    a21 = 0;
//    a22 = exp_val;

    d_diffusion_tensor[y*tensor_stride+x] = (1.0f/mag_grad)*make_float4(a11,a12,a21,a22);

}


void buildDiffusionTensor(float *d_ref_image, float* d_u0, const unsigned int ref_stride, float4 *d_diffusion_tensor, const unsigned int tensor_stride,
                          const unsigned int width, const unsigned int height, const float alpha, const float beta)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_buildDiffusionTensor<<<grid,block>>>(d_ref_image,
                                                d_u0,
                                          ref_stride,
                                          d_diffusion_tensor,
                                          tensor_stride,
                                          width,
                                          height,
                                          alpha,
                                          beta);

}


__global__ void kernel_doExactSearch(  float* d_ref_image,
                                       float* d_u,
                                       float* d_u0,
                                       const unsigned int width,
                                       const unsigned int height,
                                       const unsigned int stride,
                                       const float lambda,
                                       const float theta
                                     )
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float min_val=9999.0f;
    int u_update = 0 ;

    int hf_window_width = 4;

    for(int u0 = -hf_window_width ; u0 <= hf_window_width; u0++)
    {
        if ( x + u0 >= 0 && x+u0 < width)
        {
            float diff     = d_u[y*stride+x]-u0;
            float pix_diff = (tex2D(TexImgCur,x+u0+0.5,y+0.5) - d_ref_image[y*stride+x]);

            float data_term_val = 0.5*theta*diff*diff + lambda*abs(pix_diff);

            if ( data_term_val < min_val)
            {
                min_val = data_term_val;
                u_update = u0;
            }
        }
    }

    d_u0[y*stride+x] = u_update;
}



void exactSearch(float* d_ref_image,
                 float* d_u,
                 float* d_u0,
                 const unsigned int width,
                 const unsigned int height,
                 const unsigned int stride,
                 const float lambda,
                 const float theta)
{
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doExactSearch<<<grid,block>>>(d_ref_image,
                                         d_u,
                                         d_u0,
                                         width,
                                         height,
                                         stride,
                                         lambda,
                                         theta
                                          );
}
