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


__global__ void kernel_doOneIterationUpdatePrimal ( float* d_u,
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
                                                   const float sigma_p)
{


    /// Update Equations should be
    /// u = u - tau*( lambda*q*grad - divp )

    float dxp = 0 , dyp = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x >= 1 && x < width  )  dxp = d_px[y*stride+x] - d_px[y*stride+(x-1)];

    if ( y >= 1 && y < height )  dyp = d_py[y*stride+x] - d_py[(y-1)*stride+x];

    float div_p = dxp + dyp;

//    float u_update = d_u[y*stride+x] + sigma_u*div_p - sigma_u*lambda*d_q[y*stride+x]*d_gradient_term[y*stride+x];

//    //d_u[y*stride+x] = u_update;

//    d_u[y*stride+x] = fmaxf(0.0f,fminf(1.0f,u_update));


//    float grad_sqr = d_gradient_term[y*stride+x]*d_gradient_term[y*stride+x];

//    float u_ = (d_u[y*stride+x] + sigma_u*(div_p));

//    float u0 = d_u0[y*stride+x];

//    float rho = d_data_term[y*stride+x] + (u_-u0)*d_gradient_term[y*stride+x];

//    if ( rho < -sigma_u*lambda*grad_sqr)

//        d_u[y*stride+x] =  u_ + sigma_u*lambda*d_gradient_term[y*stride+x];

//    else if( rho > sigma_u*lambda*grad_sqr)

//        d_u[y*stride+x] =  u_ - sigma_u*lambda*d_gradient_term[y*stride+x];

//    else if ( fabs(rho) <= sigma_u*lambda*grad_sqr)
//        d_u[y*stride+x] =  u_ - rho/(d_gradient_term[y*stride+x]+10E-6);


    float grad = d_data_term[y*data_stride+x].z;
    float a_b  = (d_data_term[y*data_stride+x].x - d_data_term[y*data_stride+x].y);

//    int patch_hf_width = 1;

//    int count = 0;

//    float mu_cur=0,mu_ref=0;

//    for(int i = -patch_hf_width ; i <= patch_hf_width ; i++ )
//    {
//        for(int j = -patch_hf_width ; j <= patch_hf_width ; j++ )
//        {
//            if ( x+i >= 0 && x+i<= width && y+j >=0 && y+j <= height)
//            {
//                /// x is the cur data, y is the ref data
//                float Id = d_data_term[(y+j)*data_stride+x+i].x;
//                float Ir = d_data_term[(y+j)*data_stride+x+i].y;
//                mu_cur += Id;
//                mu_ref += Ir;
//                count++;
//            }
//        }
//    }

//    mu_cur  = mu_cur/(float)count;
//    mu_ref  = mu_ref/(float)count;


////    printf("mu_cur = %f, mu_ref = %f",mu_cur,mu_ref);
//    for(int i = -patch_hf_width ; i <= patch_hf_width ; i++ )
//    {
//        for(int j = -patch_hf_width ; j <= patch_hf_width ; j++ )
//        {
//            if ( x+i >= 0 && x+i<= width && y+j >=0 && y+j <= height)
//            {
//                /// x is the cur data, y is the ref data
//                float Id = d_data_term[(y+j)*data_stride+(x+i)].x;
//                float Ir = d_data_term[(y+j)*data_stride+(x+i)].y;

////                a_b += (Id-(mu_cur/mu_ref)*Ir);
//                a_b += (Id-(mu_cur/mu_ref)*Ir);

//                /// z is the gradient
//                grad += d_data_term[(y+j)*data_stride+x+i].z;
//            }
//        }
//    }

//    a_b = a_b ;// / (float)count;
//    grad = grad;// / (float)count;

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



        //d_u[y*stride+x] = fmaxf(0.0f,fminf(1.0f,d_u[y*stride+x]));

    //    float diff_term = d_q[y*stride+x]*d_gradient_term[y*stride+x] - div_p;




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
                                 const float sigma_p)
{

    dim3 block(boost::math::gcd<unsigned>(width,8), boost::math::gcd<unsigned>(height,8), 1);
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
                                                       sigma_p);



}




__global__ void kernel_doOneIterationUpdateDualData( float* d_q,
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
                                             const float sigma_p)
{

    /// Update Equations should be
    /// q = q - sigma_q*( lambda*data_term )
    /// q = q / max(1.0f,fabs(q))

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

//    float epsilon=0.001;
    float u = d_u[y*stride+x];
    float u0 = d_u0[y*stride+x];

    float q_update = d_q[y*stride+x] + sigma_q*lambda*(d_data_term[y*stride+x]+ (u - u0)*d_gradient_term[y*stride+x]);

//    q_update = q_update/(1+epsilon*sigma_q);

    float reprojection_q = max(1.0f,fabs(q_update)/*/lambda*/);

    d_q[y*stride+x] = q_update/reprojection_q;



}



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







__global__ void kernel_doOneIterationUpdateDualReg (float* d_px,
                                                    float* d_py,
                                                    float* d_u,
                                                    const unsigned int stride,
                                                    const unsigned int width,
                                                    const unsigned int height,
                                                    const float lambda,
                                                    const float sigma_u,
                                                    const float sigma_q,
                                                    const float sigma_p)
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

    float pxval = d_px[y*stride+x] + sigma_p*(u_dx);
    float pyval = d_py[y*stride+x] + sigma_p*(u_dy);

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
                                  const float sigma_p)
{

    dim3 block(boost::math::gcd<unsigned>(width,8), boost::math::gcd<unsigned>(height,8), 1);
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
                                                       sigma_p);

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

//        Id_minus_Ir = Id-Ir;


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

    dim3 block(boost::math::gcd<unsigned>(width,8), boost::math::gcd<unsigned>(height,8), 1);
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

    dim3 block(boost::math::gcd<unsigned>(width,8), boost::math::gcd<unsigned>(height,8), 1);
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







