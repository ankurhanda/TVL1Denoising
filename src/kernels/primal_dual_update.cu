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
#include <thrust/sort.h>
#include <thrust/pair.h>


texture<float, 2, cudaReadModeElementType> TexImgCur;

const static cudaChannelFormatDesc chandesc_float1 =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

texture<float, 3, cudaReadModeElementType> TexImgStack;



__global__ void kernel_doOneIterationUpdatePrimal ( float* d_u,
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
                                                   const int _nimages)
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

//    d_u[y*stride+x] = u_update;

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



    /// Have a confusion of _nimages

//    float u0  = d_u0[y*stride+x];
//    float u_  = d_u[y*stride+x] + sigma_u*(div_p);
//    float ai, bi, ti;

//    float *ti_vals = new float [_nimages-1];
//    unsigned int *ti_indices = new unsigned int [_nimages-1];
//    float sum_all_grads = 0;

//    for(int i = 0 ; i < _nimages -1 ; i++)
//    {

//        bi = d_data_term[y*stride+x+i*slice_stride] - u0*d_gradient_term[y*stride+x+i*slice_stride];
//        ai = d_gradient_term[y*stride+x+i*slice_stride];
//        ti = -bi/(ai+1E-6);

//        ti_vals[i]=ti;
//        ti_indices[i]=i;

//        sum_all_grads += ai;
//    }

    /// Sort them.
//    thrust::stable_sort_by_key(ti_vals,ti_vals+_nimages-1,ti_indices);



    /// Find if this lies in between any of these consecutive ti s.
//    float sum_grads_less_k = 0;
//    float sum_grads_great_k = sum_all_grads;
//    int index = 0;
//    bool found_min = false;
//    float rho=0;

//    for(int i = 1 ; i <_nimages - 1 ; i++)
//    {
//        index = ti_indices[i];

//        rho = d_data_term[y*stride+x+index*slice_stride] + (u_-u0)*d_gradient_term[y*stride+x+index*slice_stride];

//        sum_grads_less_k  += d_gradient_term[y*stride+x+index*slice_stride];
//        sum_grads_great_k -= sum_grads_less_k;

//        if (  rho < sigma_u*lambda*(sum_grads_less_k- sum_grads_great_k) && rho > d_gradient_term[y*stride+x+index*slice_stride]*(ti_vals[i-1] - ti_vals[i]) )
//        {
//            d_u[y*stride] = u_ + sigma_u*lambda*(sum_grads_less_k - sum_grads_great_k);
//            found_min = true;
//            return;
//        }
//    }

//    if ( !found_min )
//    {
//        /// Bound check at 0
//        index = ti_indices[0];
//        rho = d_data_term[y*stride+index*slice_stride] + (u_-u0)*d_gradient_term[y*stride+x+index*slice_stride];
//        if ( rho < -sigma_u*lambda*sum_all_grads)
//        {
//            d_u[y*stride+x]  = u_ +  sigma_u*lambda*sum_all_grads;
//            return;
//        }

//        /// Bound check at last
//        index = ti_indices[_nimages-2];
//        rho = d_data_term[y*stride+index*slice_stride] + (u_-u0)*d_gradient_term[y*stride+x+index*slice_stride];
//        if ( rho > sigma_u*lambda*sum_all_grads)
//        {
//            d_u[y*stride+x]  = u_ -  sigma_u*lambda*sum_all_grads;
//            return;
//        }


//        /// Check for minima among the ti points
//        float cur_min_cost = 1E20;
//        int min_di_index = 0;

//        for(int i = 0 ; i < _nimages-1 ; i++)
//        {
//            float di = ti_vals[i];
//            float sum_rhos_at_di = 0;

//            for(int j = 0 ; j < _nimages -1 ; j++)
//            {
//                bi = d_data_term[y*stride+x+j*slice_stride] - u0*d_gradient_term[y*stride+x+j*slice_stride];
//                ai = d_gradient_term[y*stride+x+j*slice_stride];

//                sum_rhos_at_di += fabs(ai*di-bi);
//            }

//           float min_cost = -div_p * di + lambda * sum_rhos_at_di;

//           if ( min_cost < cur_min_cost)
//           {
//               cur_min_cost = min_cost;
//               min_di_index = i;
//           }

//        }

//        float rho_u = d_data_term[y*stride+x+min_di_index*slice_stride] + (u_- u0)*d_gradient_term[y*stride+x+min_di_index*slice_stride];
//        float gradient_at_u = d_gradient_term[y*stride+x+min_di_index*slice_stride];

//        d_u[y*stride+x] = u_ - rho_u/(gradient_at_u+1E-6);

//        return;

//    }

//        d_u[y*stride+x] = fmaxf(1E-6,fminf(1.0f,d_u[y*stride+x]));
//        float diff_term = d_q[y*stride+x]*d_gradient_term[y*stride+x] - div_p;




}

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
                                 const int _nimages)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdatePrimal<<<grid,block>>>( d_u,
                                                       d_u0,
                                                       stride,
                                                       width,
                                                       height,
                                                       d_data_term,
                                                       d_gradient_term,
                                                       d_px,
                                                       d_py,
                                                       d_q,
                                                       lambda,
                                                       sigma_u,
                                                       sigma_q,
                                                       sigma_p,
                                                      _nimages);


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

    float reprojection_q = max(1.0f,fabs(q_update));

    d_q[y*stride+x] = q_update/reprojection_q;



}



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
                                  const float sigma_p)
{

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    kernel_doOneIterationUpdateDualData<<<grid,block>>>(d_q,
                                                       stride,
                                                       width,
                                                       height,
                                                       d_data_term,
                                                       d_gradient_term,
                                                       d_u,
                                                       d_u0,
                                                       lambda,
                                                       sigma_u,
                                                       sigma_q,
                                                       sigma_p);

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
    float reprojection_p   = fmaxf(1.0f, length( make_float2(pxval,pyval) ) );

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
                                                       sigma_p);

}



__global__ void kernel_computeImageGradient_wrt_depth(const float2 fl,
                                               const float2 pp,
                                               float* d_u,
                                               float* d_u0,
                                               float* d_data_term,
                                               float* d_gradient_term,
                                               cumat<3,3>R,
                                               cumat<3,1>t,
                                               const unsigned int stride,
                                               float* d_ref_img,
                                               const unsigned int width,
                                               const unsigned int height,
                                               bool disparity,
                                               float dmin,
                                               float dmax,
                                               const unsigned int which_image,
                                               const unsigned int slice_stride)
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

//        float u0 = d_u0[y*stride+x];
//        float u  = d_u[y*stride+x];

        d_gradient_term[y*stride+x] = grad_I2_u0 +10E-6;

        float data_term_value  = (I2_u0 /*+ (u-u0)*grad_I2_u0 */- I1_val);

        d_data_term[y*stride+x] = data_term_value;
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
        zLinearised = fmaxf(1E-6,fminf(1.0f,zLinearised));

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


//        float Id =  tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f);
//        float Idx = tex2D(TexImgCur,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);

        float Id =  tex3D(TexImgStack, p2D_live.x+0.5f,     p2D_live.y+0.5f,which_image);
        float Idx = tex3D(TexImgStack, p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f,which_image);


//        if ( p2D_live.x+0.5+1 > (float) width)
//            Idx = Id;

//        float Idy = tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);
        float Idy = tex3D(TexImgStack,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f, which_image);

//        if ( p2D_live.y+0.5+1 > (float) height)
//            Idy = Id;


        float2 dIdx = make_float2(Idx-Id, Idy-Id);

        p3d_dest_vec.z = p3d_dest_vec.z + 10E-6;

//        float3 dpi_u = make_float3(1/p3d_dest_vec.z, 0,-(p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
//        float3 dpi_v = make_float3(0, 1/p3d_dest_vec.z,-(p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        float3 dpi_u = make_float3(fl.x/p3d_dest_vec.z, 0,-(fl.x*p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
        float3 dpi_v = make_float3(0, fl.y/p3d_dest_vec.z,-(fl.y*p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        cumat<3,1> dXdz = R*uvnormMat; ///(-zLinearised*zLinearised);

        float3 dXdz_vec = {dXdz(0,0),dXdz(1,0),dXdz(2,0)};

        dIdz =  dot(dIdx, make_float2( dot(dXdz_vec,dpi_u),  dot(dXdz_vec,dpi_v) ) );

        Id_minus_Ir = Id-Ir;

        d_data_term[y*stride+x + which_image*slice_stride] = Id_minus_Ir ;//+ (u-u0)*dIdz;
        d_gradient_term[y*stride+x + which_image*slice_stride] = dIdz;

    }

}


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
                                    const unsigned int slice_stride)
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
                                          d_gradient_term,
                                          R,
                                          t,
                                          stride,
                                          d_ref_img,
                                          width,
                                          height,
                                          disparity,
                                          dmin,
                                          dmax,
                                          which_image,
                                          slice_stride);
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





void BindDepthTexture(float* cur_img,
                      unsigned int width,
                      unsigned int height,
                      unsigned int imgStride)

{
    cudaBindTexture2D(0,TexImgCur,cur_img,chandesc_float1,width,height,imgStride*sizeof(float));

    TexImgCur.addressMode[0] = cudaAddressModeClamp;
    TexImgCur.addressMode[1] = cudaAddressModeClamp;
    TexImgCur.filterMode = cudaFilterModeLinear;
    TexImgCur.normalized = false;    // access with normalized texture coordinates
}


void BindDataImageStack ( const cudaArray *d_volumeArray,
                          const unsigned int width,
                          const unsigned int height,
                          const unsigned int depth,
                          cudaChannelFormatDesc channelDesc)
{
    /// Bind array to 3D texture
    cutilSafeCall(cudaBindTextureToArray(TexImgStack, d_volumeArray, channelDesc));

    /// Set Texture Parameters
    TexImgStack.normalized = false;                      // Access with normalized texture coordinates
    TexImgStack.filterMode = cudaFilterModeLinear;       // Linear interpolation
    TexImgStack.addressMode[0] = cudaAddressModeClamp;   // Clamp texture coordinates
    TexImgStack.addressMode[1] = cudaAddressModeClamp;
    TexImgStack.addressMode[2] = cudaAddressModeClamp;

}




