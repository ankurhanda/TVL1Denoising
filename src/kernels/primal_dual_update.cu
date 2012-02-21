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


__global__ void kernel_doOneIterationUpdatePrimal ( float* d_primal,
                                                   const unsigned int stride,
                                                   const unsigned int width,
                                                   const unsigned int height,
                                                   const float* d_gradient_term,
                                                   const float2* d_dual_reg,
                                                   const float* d_dual_data,
                                                   const float lambda,
                                                   const float sigma_primal,
                                                   const float sigma_dual_data,
                                                   const float sigma_dual_reg)
{

    /// Update Equations should be
    /// u = u - tau*( lambda*q*grad - divp )

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float dxp = 0, dyp = 0;

    if ( x >= 1 && x < width)
         dxp = d_dual_reg[y*stride+x].x - d_dual_reg[y*stride+(x-1)].x ;

    if ( y >= 1 && y < height)
         dyp = d_dual_reg[y*stride+x].y - d_dual_reg[(y-1)*stride+x].y ;

    float div_p = dxp + dyp;
    float primal_update = lambda*d_dual_data[y*stride+x]*d_gradient_term[y*stride+x] - div_p;

    d_primal[y*stride+x] = d_primal[y*stride+x] - sigma_primal*primal_update;





}

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
                                   const float sigma_dual_reg)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdatePrimal<<<grid,block>>>(d_primal,
                                                       stride,
                                                       width,
                                                       height,
                                                       d_gradient_term,
                                                       d_dual_reg,
                                                       d_dual_data,
                                                       lambda,
                                                       sigma_primal,
                                                       sigma_dual_data,
                                                       sigma_dual_reg);



}




__global__ void kernel_doOneIterationUpdateDualData( float* d_dual_data,
                                             const unsigned int stride,
                                             const unsigned int width,
                                             const unsigned int height,
                                             const float* d_data_term,
                                             const float lambda,
                                             const float sigma_primal,
                                             const float sigma_dual_data,
                                             const float sigma_dual_reg)
{

    /// Update Equations should be
    /// q = q - sigma_q*( data_term )
    /// q = q / max(1.0f,fabs(q))

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float dual_data_update = d_dual_data[y*stride+x] + sigma_dual_data * d_data_term[y*stride+x];

    float reprojection     = fmaxf(1.0f, fabs(dual_data_update));

    d_dual_data[y*stride+x] = dual_data_update / reprojection;



}



void doOneIterationUpdateDualData(float* d_dual_data,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float* d_data_term,
                                 const float lambda,
                                 const float sigma_primal,
                                 const float sigma_dual_data,
                                 const float sigma_dual_reg)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdateDualData<<<grid,block>>>(d_dual_data,
                                                       stride,
                                                       width,
                                                       height,
                                                       d_data_term,
                                                       lambda,
                                                       sigma_primal,
                                                       sigma_dual_data,
                                                       sigma_dual_reg);

}







__global__ void kernel_doOneIterationUpdateDualReg (float2* d_dual_reg,
                                                    float* d_primal,
                                                    const unsigned int stride,
                                                    const unsigned int width,
                                                    const unsigned int height,
                                                    const float lambda,
                                                    const float sigma_primal,
                                                    const float sigma_dual_data,
                                                    const float sigma_dual_reg)
{


    /// Update Equations should be
    /// p = p - sigma_p*( grad_d )
    /// p = p / max(1.0f,length(p))

    float u_dx = 0, u_dy = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x + 1 < width )
    {
        u_dx = d_primal[y*stride+(x+1)] - d_primal[y*stride+x];
    }

    if ( y + 1 < height )
    {
        u_dy = d_primal[(y+1)*stride+x] - d_primal[y*stride+x];
    }

    float pxval = d_dual_reg[y*stride+x].x + sigma_dual_reg*(u_dx)*lambda;
    float pyval = d_dual_reg[y*stride+x].y + sigma_dual_reg*(u_dy)*lambda;

    // reprojection
    float reprojection_p   = fmaxf(1.0f,length( make_float2(pxval,pyval) ));
    d_dual_reg[y*stride+x] = make_float2  ( pxval/reprojection_p, pyval/reprojection_p);


}



void doOneIterationUpdateDualReg(float2* d_dual_reg,
                                 float* d_primal,
                                 const unsigned int stride,
                                 const unsigned int width,
                                 const unsigned int height,
                                 const float lambda,
                                 const float sigma_primal,
                                 const float sigma_dual_data,
                                 const float sigma_dual_reg)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdateDualReg<<<grid,block>>>(d_dual_reg,
                                                       d_primal,
                                                       stride,
                                                       width,
                                                       height,
                                                       lambda,
                                                       sigma_primal,
                                                       sigma_dual_data,
                                                       sigma_dual_reg);

}



__global__ void kernel_computeImageGradient_wrt_depth(const float2 fl,
                                               const float2 pp,
                                               float* d_primal,
                                               float* d_primal_u0,
                                               float* d_data_term,
                                               float* d_gradient_term,
                                               cumat<3,3>R,
                                               cumat<3,1>t,
                                               const unsigned int stride,
                                               float* d_ref_img,
                                               const unsigned int width,
                                               const unsigned int height)
{


    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    /// Checked!
    float2 invfl = 1.0f/fl;

    /// Checked!
    float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);//*(dmax-dmin);

    float zLinearised = d_primal_u0[y*stride+x];

    cumat<3,1> p3d       = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, zLinearised};
    cumat<3,1> uvnormMat = {uvnorm.x, uvnorm.y, uvnorm.z};
    cumat<3,1> p3d_dest  =  R*p3d + t;

    float dIdz;
    float Id_minus_Ir;

    float3 p3d_dest_vec = {p3d_dest(0,0), p3d_dest(1,0), p3d_dest(2,0)};

    float2 p2D_live = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

    p2D_live.x= p2D_live.x*fl.x + pp.x;
    p2D_live.y= p2D_live.y*fl.y + pp.y;

    float Ir =  d_ref_img[y*stride+x];

    float Id =  tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f);
    float Idx = tex2D(TexImgCur,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);

    if ( p2D_live.x+0.5+1 > (float) width)
        Idx = Id;

    float Idy = tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);

    if ( p2D_live.y+0.5+1 > (float) height)
        Idy = Id;


    float2 dIdx = make_float2(Idx-Id, Idy-Id);

    float3 dpi_u = make_float3(fl.x/p3d_dest_vec.z, 0,-(fl.x*p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
    float3 dpi_v = make_float3(0, fl.y/p3d_dest_vec.z,-(fl.y*p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

    cumat<3,1> dXdz = R*uvnormMat;
    float3 dXdz_vec = {dXdz(0,0),dXdz(1,0),dXdz(2,0)};

    dIdz =  dot(dIdx, make_float2( dot(dXdz_vec,dpi_u),  dot(dXdz_vec,dpi_v) ) );

    Id_minus_Ir = Id-Ir;


    float u  =  d_primal[y*stride+x];
    float u0 = d_primal_u0[y*stride+x];

    d_data_term[y*stride+x] = Id_minus_Ir + (u-u0)*dIdz;
    d_gradient_term[y*stride+x] = dIdz;

}


void doComputeImageGradient_wrt_depth(const float2 fl,
                                    const float2 pp,
                                    float* d_primal,
                                    float* d_primal_u0,
                                    float* d_data_term,
                                    float* d_gradient_term,
                                    TooN::Matrix<3,3>R_lr_,
                                    TooN::Matrix<3,1>t_lr_,
                                    const unsigned int stride,
                                    float* d_ref_img,
                                    const unsigned int width,
                                    const unsigned int height)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    cumat<3,1> t = cumat_from<3,1,float>(t_lr_);

    kernel_computeImageGradient_wrt_depth<<<grid,block>>>(fl,
                                          pp,
                                          d_primal,
                                          d_primal_u0,
                                          d_data_term,
                                          d_gradient_term,
                                          R,
                                          t,
                                          stride,
                                          d_ref_img,
                                          width,
                                          height);
}








__global__ void kernel_doImageWarping( const float2 fl,
                                       const float2 pp,
                                       const cumat<3,3> R,
                                       const cumat<3,1> t,
                                       float* d_cur2ref_warped,
                                       float* d_primal,
                                       const unsigned int stride,
                                       const unsigned int width,
                                       const unsigned int height)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float2 invfl = 1.0f/fl;
    float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);

    float zLinearised = d_primal[y*stride+x];

    cumat<3,1> p3d      = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, uvnorm.z*zLinearised};
    cumat<3,1> p3d_dest = R*p3d + t;
    float2 p2D_live     = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

    p2D_live.x = p2D_live.x*fl.x + pp.x;
    p2D_live.y = p2D_live.y*fl.y + pp.y;

    d_cur2ref_warped[y*stride+x] = tex2D(TexImgCur,p2D_live.x,p2D_live.y);
}



void doImageWarping(const float2 fl,
                    const float2 pp,
                    TooN::Matrix<3,3> R_lr_,
                    TooN::Matrix<3,1> t_lr_,
                    float *d_cur2ref_warped,
                    float *d_primal,
                    const unsigned int stride,
                    const unsigned int width,
                    const unsigned int height)
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
                                          d_primal,
                                          stride,
                                          width,
                                          height);

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







