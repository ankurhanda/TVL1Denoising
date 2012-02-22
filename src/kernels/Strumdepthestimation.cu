

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
#include "Strumdepthestimation.h"
#include <iostream>


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

texture<float, 2, cudaReadModeElementType> TexImg2;

const static cudaChannelFormatDesc chandesc_float1 =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);


__global__ void depth_estimation_kernel_q (float* dq, const float* __restrict__ du, const float* __restrict__ du0,
                                           float sigma_q, float lambda, float *dI1,
                                           unsigned int width, unsigned int height, unsigned int stride,
                                           float *data_term)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


    float q_update = dq[y*stride+x] + sigma_q*data_term[y*stride+x]/**lambda*/;

    float reprojection_q = max(1.0f,fabs(q_update));

    dq[y*stride+x] = q_update/reprojection_q;


}

void launch_depth_estimation_kernel_q (float* dq, float *du, float *du0,
                                       float sigma_q, float lambda, float *dI1,
                                       unsigned int width, unsigned int height,
                                       unsigned int stride, float *data_term)
{
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    depth_estimation_kernel_q <<<grid,block>>> (dq,du,du0,sigma_q,
                                                lambda,dI1,width,
                                                height,stride,
                                                data_term);
}


__global__ void depth_estimation_kernel_u(const float* __restrict__ dpx, const float* __restrict__ dpy,float *du, float *du0, float *dq, float *dI1,
                                   float sigma_u, float lambda, unsigned int width,
                                   unsigned height, unsigned int stride, /*const cumat3x3 KRKinv, cumat<3,1> cudaKt,*/
                                         float dmin, float dmax, float *grad_wrt_d_at_d0)
{
    float dxp = 0 , dyp = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x >= 1 && x < width  )  dxp = dpx[y*stride+x] - dpx[y*stride+(x-1)];

    if ( y >= 1 && y < height ) dyp = dpy[y*stride+x] - dpy[(y-1)*stride+x];

    float div_p = dxp + dyp;

    float diff_term = /*lambda**/dq[y*stride+x]*grad_wrt_d_at_d0[y*stride+x] - div_p;

    float u_ = (du[y*stride+x] - sigma_u*(diff_term));
//    du[y*stride+x] = u_;
//    u_ = fmaxf(10E-6,fminf(1,u_));
    du[y*stride+x] =  u_;
//  du[y*stride+x] = fminf(1.0f,fmaxf(0.0f,du[y*stride+x]));

}


void launch_depth_estimation_kernel_u(float* dpx, float* dpy,float *du, float *du0,float *dq, float *dI1,
                               float sigma_u, float lambda, unsigned int width,
                               unsigned int height, unsigned int stride, /* const TooN::Matrix<3,3>& K,
                               const TooN::Matrix<3,3>& Kinv, const TooN::Matrix<3,3>& R_lr, const TooN::Matrix<3,1>& Kt,*/
                                      float dmin, float dmax, float *grad_wrt_d_at_d0)
{
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    depth_estimation_kernel_u<<<grid,block>>>(dpx,dpy,du,du0,dq,dI1,sigma_u,lambda,width,height,stride,
                                              dmin, dmax,
                                              grad_wrt_d_at_d0);

}


__global__ void depth_estimation_kernel_p(float* dpx, float *dpy, const float* __restrict__ du,
                                   unsigned int width, unsigned int height, unsigned int stride,
                                   float lambda,
                                   float sigma_p, float dmin, float dmax)

{
    float u_dx = 0, u_dy = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x + 1 < width )
    {
        u_dx = du[y*stride+(x+1)] - du[y*stride+x];
    }

    if ( y + 1 < height )
    {
        u_dy = du[(y+1)*stride+x] - du[y*stride+x];
    }

//    float2 update = (p_n + du_n*tau); ///(1+huber_eps*tau);
//    const float len =  fmaxf(1,length(update)*lambda);

//    d_px[indexf1] =  update.x/len;
//    d_py[indexf1] =  update.y/len;


    float pxval = dpx[y*stride+x] + sigma_p*(u_dx)*lambda;
    float pyval = dpy[y*stride+x] + sigma_p*(u_dy)*lambda;

    // reprojection
    float reprojection_p   = fmaxf(1.0f,sqrtf(pxval*pxval + pyval*pyval));

    dpx[y*stride+x] = pxval / reprojection_p;
    dpy[y*stride+x] = pyval / reprojection_p;
}


void launch_depth_estimation_kernel_p(float* dpx, float *dpy, float *du,
                                          unsigned int width, unsigned int height, unsigned int stride,
                                       float lambda,
                                          float sigma_p, float dmin, float dmax)
{
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);
    depth_estimation_kernel_p<<<grid, block>>>(dpx, dpy, du, width,height,stride,lambda,sigma_p, dmin, dmax);

}



__global__ void depth_estimation_I2warped(float* dI2warped,float* __restrict__ du,unsigned int width,
                                      unsigned int height,unsigned int stride,
                                          const float2 fl,
                                          const float2 pp,
                                          const cumat<3,3> R,
                                          const cumat<3,1> t,
//                                         const cumat3x3 KRKinv,
//                                          cumat<3,1> cudaKt,
                                          float dmin,
                                          float dmax)
{


//    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

////    float uval = (du[y*stride+x]-dmin)/(dmax-dmin);
////    float uval = (du[y*stride+x]-dmin)/(dmax-dmin);



//    const cumat<3,1> p0 = {x*uval, y*uval,uval};
//    const cumat<3,1> p2 = KRKinv*p0 + cudaKt;
//    const float2 pIn2   = {p2(0,0)/p2(2,0), p2(1,0)/p2(2,0)};

//    dI2warped[y*stride+x] = tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5);


    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float2 invfl = 1.0f/fl;
    float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);

    float zLinearised = du[y*stride+x];

    cumat<3,1> p3d      = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, uvnorm.z*zLinearised};
    cumat<3,1> p3d_dest = R*p3d + t;
    float2 p2D_live = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

    p2D_live.x = p2D_live.x*fl.x + pp.x;
    p2D_live.y = p2D_live.y*fl.y + pp.y;

    dI2warped[y*stride+x] = tex2D(TexImg2,p2D_live.x,p2D_live.y);

}


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
                                             float dmax)
{
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    cumat<3,1> t = cumat_from<3,1,float>(t_lr_);

    //    cumat3x3   KRKinv = cumat_from<3,3,float>(K*R_lr*Kinv);
    //    cumat<3,1> cudaKt = cumat_from<3,1,float>(Kt);

    depth_estimation_I2warped<<<grid, block>>>(dI2warped,du,width,
                                               height,stride,/*KRKinv,cudaKt,*/
                                               fl,
                                               pp,
                                               R,
                                               t,
                                               dmin,
                                               dmax);

}


__global__ void depth_estimation_kernel_compute_data_term_and_gradient(const float *dI1,
                                                                       float *data_term,
                                                                       float *grad_wrt_d_at_d0,
                                                                       const float* __restrict__ du,const float*  __restrict__ du0,
                                                                       const unsigned int height,
                                                                       const unsigned int width,
                                                                       const unsigned int stride,
                                                                       const float2 fl,
                                                                       const float2 pp,
                                                                       const cumat<3,3> R,
                                                                       const cumat<3,1> t,
                                                                       float dmin,
                                                                       float dmax)
{


//    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


//    float xinterp0 = (float)x+ du0[y*stride+x];
//    float xinterp1 = (float)x+ du0[y*stride+x]+1;


//    float I2_u0      = tex2D(TexImg2,xinterp0+0.5,(float)y+0.5);
//    float I1_val     = dI1[y*stride+x];
//    float grad_I2_u0 = tex2D(TexImg2,xinterp1+0.5,(float)y+0.5) - tex2D(TexImg2,xinterp0+0.5,(float)y+0.5);

//    float u0 = du0[y*stride+x];
//    float u  = du[y*stride+x];

//    grad_wrt_d_at_d0[y*stride+x] = grad_I2_u0;

//    float data_term_value  = (I2_u0 + (u-u0)*grad_I2_u0 - I1_val);

//    data_term[y*stride+x] = data_term_value;


    bool usemine = false;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    /// Checked!
    float2 invfl = 1.0f/fl;

    /// Checked!
    float3 uvnorm =  make_float3( (x-pp.x)*invfl.x, (y-pp.y)*invfl.y,1);//*(dmax-dmin);

    float zLinearised = du0[y*stride+x];

    cumat<3,1> p3d       = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, zLinearised};
    cumat<3,1> uvnormMat = {uvnorm.x, uvnorm.y, uvnorm.z};
    cumat<3,1> p3d_dest  =  R*p3d + t;

    float dIdz;
    float Id_minus_Ir;

    float3 p3d_dest_vec = {p3d_dest(0,0), p3d_dest(1,0), p3d_dest(2,0)};


    if ( usemine )
    {
        float2 p2D_live = {fl.x*(p3d_dest_vec.x/p3d_dest_vec.z) + pp.x, fl.y*(p3d_dest_vec.y/p3d_dest_vec.z) + pp.y};

        p2D_live.x = fmax(0, fmin(width,p2D_live.x));
        p2D_live.y = fmax(0,fmin(height,p2D_live.y));

        float Ir = dI1[y*stride+x];

        float Id =  tex2D(TexImg2,  p2D_live.x+0.5f,p2D_live.y+0.5f);

        float Idx = tex2D(TexImg2,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);

        if ( p2D_live.x+0.5+1 > (float) width)
            Idx = Id;

        float Idy = tex2D(TexImg2,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);

        if ( p2D_live.y+0.5+1 > (float) height)
            Idy = Id;


        float2 dIdx = make_float2(Idx-Id, Idy-Id);

//        p3d_dest_vec.z = p3d_dest_vec.z + 0.001;

        float3 dpi_u = make_float3(1.0f/(p3d_dest_vec.z), 0,  -(p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
        float3 dpi_v = make_float3(0, 1.0f/p3d_dest_vec.z,  -(p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        cumat<3,1> RKinvpoint  = R*uvnormMat;///(-zLinearised*zLinearised);
        float3 KRKinvpoint_vec = {fl.x*RKinvpoint(0,0) + pp.x, fl.y*RKinvpoint(1,0) + pp.y, RKinvpoint(2,0)};

        dIdz = dot(dIdx, make_float2( dot(dpi_u,KRKinvpoint_vec), dot(dpi_v,KRKinvpoint_vec)) );

        Id_minus_Ir = Id-Ir;

    }
    else
    {

        float2 p2D_live = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

        p2D_live.x= p2D_live.x*fl.x + pp.x;
        p2D_live.y= p2D_live.y*fl.y + pp.y;

        float Ir =  dI1[y*stride+x];

        float Id =  tex2D(TexImg2,  p2D_live.x+0.5f,p2D_live.y+0.5f);
        float Idx = tex2D(TexImg2,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);

        if ( p2D_live.x+0.5+1 > (float) width)
            Idx = Id;

        float Idy = tex2D(TexImg2,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);

        if ( p2D_live.y+0.5+1 > (float) height)
            Idy = Id;


        float2 dIdx = make_float2(Idx-Id, Idy-Id);

        float3 dpi_u = make_float3(fl.x/p3d_dest_vec.z, 0,-(fl.x*p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
        float3 dpi_v = make_float3(0, fl.y/p3d_dest_vec.z,-(fl.y*p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

        cumat<3,1> dXdz = R*uvnormMat;
        float3 dXdz_vec = {dXdz(0,0),dXdz(1,0),dXdz(2,0)};

        dIdz =  dot(dIdx, make_float2( dot(dXdz_vec,dpi_u),  dot(dXdz_vec,dpi_v) ) );

        Id_minus_Ir = Id-Ir;
    }



    float u  =  du[y*stride+x];
    float u0 = du0[y*stride+x];

    data_term[y*stride+x] = Id_minus_Ir + (u-u0)*dIdz;
    grad_wrt_d_at_d0[y*stride+x] = dIdz;

}




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
                                                                   float dmax)
{


    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);



    const cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    const cumat<3,1> t = cumat_from<3,1,float>(t_lr_);



    depth_estimation_kernel_compute_data_term_and_gradient<<<grid, block>>>(dI1,data_term,
                                                                            grad_wrt_d_at_d0,
                                                                            du, du0,
                                                                            height,
                                                                            width,
                                                                            stride,
                                                                            fl,
                                                                            pp,
                                                                            R,
                                                                            t,
//                                                                            KRKinv,
//                                                                            cudaKt,
                                                                            dmin,
                                                                            dmax);

}

void launch_depth_kernel(float* dI2, unsigned int width, unsigned int height, unsigned int imgStride)

{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    cudaBindTexture2D(0,TexImg2,dI2,chandesc_float1,width, height,imgStride*sizeof(float));

    TexImg2.addressMode[0] = cudaAddressModeClamp;
    TexImg2.addressMode[1] = cudaAddressModeClamp;
    TexImg2.filterMode = cudaFilterModeLinear;
    TexImg2.normalized = false;    // access with normalized texture coordinates


}









__global__ void depth_estimation_kernel_check_KRt_is_correct(float *du0,
                                                             unsigned int height,
                                                             unsigned int width,
                                                             unsigned int stride,
                                                             const cumat3x3 KRKinv,
                                                             const cumat<3,1> cudaKt)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float uval = du0[y*stride+x];

    const cumat<3,1> p0 = {x*uval,y*uval,uval};
    const cumat<3,1> p2 = KRKinv*p0 + cudaKt;
    const float2 pIn2   = {p2(0,0)/p2(2,0), p2(1,0)/p2(2,0)};

    if ( x == 112   && y == 112 )
    {
        printf("uval = %f\n",uval);
        printf("%f %f\n",pIn2.x,pIn2.y);
    }

}


void launch_kernel_check_KRt_is_correct(float *du0,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int stride,
                                        const TooN::Matrix<3,3>& K,
                                        const TooN::Matrix<3,3>& Kinv,
                                        const TooN::Matrix<3,3>& R_lr,
                                        const TooN::Matrix<3,1>& Kt)

{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);

    cumat3x3   KRKinv = cumat_from<3,3,float>(K*R_lr*Kinv);
    cumat<3,1> cudaKt = cumat_from<3,1,float>(Kt);

    depth_estimation_kernel_check_KRt_is_correct<<<grid,block>>>(du0, height,
                                                                 width,stride,
                                                                 KRKinv,
                                                                 cudaKt);
}


__global__ void depth_estimation_kernel_check_grad_wrt_d0_is_correct(float *du0,
                                                                     const unsigned int height,
                                                                     const unsigned int width,
                                                                     const unsigned int stride,
                                                                     const cumat3x3 KRKinv,
                                                                     const cumat<3,1> cudaKt)
{


    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    const cumat<3,1> p0 = {x*du0[y*stride+x],y*du0[y*stride+x],du0[y*stride+x]};
    const cumat<3,1> p2 = KRKinv*p0 + cudaKt;
    const float2 pIn2   = {p2(0,0)/p2(2,0), p2(1,0)/p2(2,0)};

    const cumat<3,1> p0withoutdepth = {x,y,1};
    const cumat<3,1> p2withoutdepth = KRKinv*p0withoutdepth;

    float grad_Ix = tex2D(TexImg2,pIn2.x+1+0.5,pIn2.y+0.5) -tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5);
    float grad_Iy = tex2D(TexImg2,pIn2.x+0.5,pIn2.y+1+0.5) -tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5);

    float p2_wsqr     = p2(2,0)*p2(2,0);
    float third_val = -(1/p2_wsqr)*(p2(0,0)*grad_Ix + p2(1,0)*grad_Iy);

    float3 grad_wrt_pi   = {grad_Ix*(1/p2(2,0)), grad_Iy*(1/p2(2,0)), third_val};
    float3 pxyz          = {p2withoutdepth(0,0), p2withoutdepth(1,0), p2withoutdepth(2,0)};

    float grad_I2_wrt_d_at_u0 = dot(grad_wrt_pi,pxyz);

    if ( x == 123 && y == 165 )
    {
        printf("d0 = %f\n",du0[y*stride+x]);

        printf("pIn2 = %f %f\n",pIn2.x,pIn2.y);

        printf("p2withoutdepth = %f %f %f\n",p2withoutdepth(0,0),p2withoutdepth(1,0),p2withoutdepth(2,0));

        printf("grad_val = %f\n",grad_I2_wrt_d_at_u0);
    }

}


void launch_kernel_check_grad_wrt_d0_is_correct(float *du0,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int stride,
                                        const TooN::Matrix<3,3>& K,
                                        const TooN::Matrix<3,3>& Kinv,
                                        const TooN::Matrix<3,3>& R_lr,
                                        const TooN::Matrix<3,1>& Kt)

{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);

    cumat3x3   KRKinv = cumat_from<3,3,float>(K*R_lr*Kinv);
    cumat<3,1> cudaKt = cumat_from<3,1,float>(Kt);

    depth_estimation_kernel_check_grad_wrt_d0_is_correct<<<grid,block>>>(du0, height,
                                                                 width,stride,
                                                                 KRKinv,
                                                                 cudaKt);
}



__global__ void depth_estimation_kernel_copy_u0_to_u(float* du, float *du0,
                                   unsigned int width, unsigned int height, unsigned int stride)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    du0[y*stride+x] = du[y*stride+x];
}

void  launch_depth_estimation_kernel_copy_u0_to_u(float *du,float *du0, unsigned int width, unsigned int height, unsigned int stride)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    depth_estimation_kernel_copy_u0_to_u<<<grid,block>>>(du, du0, width,height,stride);
}



/*
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


//    float xinterp0 = (float)x+ du0[y*stride+x];
//    float xinterp1 = (float)x+ du0[y*stride+x]+1;


//    float I2_u0      = tex2D(TexImg2,xinterp0+0.5,(float)y+0.5);
//    float I1_val     = dI1[y*stride+x];
//    float grad_I2_u0 = tex2D(TexImg2,xinterp1+0.5,(float)y+0.5) - tex2D(TexImg2,xinterp0+0.5,(float)y+0.5);

//    float u0 = du0[y*stride+x];
//    float u  = du[y*stride+x];

//    grad_wrt_d_at_d0[y*stride+x] = grad_I2_u0;

//    float data_term_value  = (I2_u0 + (u-u0)*grad_I2_u0 - I1_val);

//    data_term[y*stride+x] = data_term_value;


/// [grad_Ix grad_Iy]*[ 1/w 0 -u/w^2 * KRK^(-1) [x
///                     0 1/w -v/w^2]            y
///                                              1];

bool dividebyrange = false;

float uval = 0;

if ( dividebyrange )
     uval = (du0[y*stride+x]-dmin)/(dmax-dmin);
else
     uval  = du0[y*stride+x];
//    uval = fminf(1.0f,fmaxf(0.0f,uval));

const cumat<3,1> p0 = {x*uval,y*uval,uval};
const cumat<3,1> p2 = KRKinv*p0 + cudaKt;

const float3 Pc2    = {p2(0,0),p2(1,0),p2(2,0)};
float2 pIn2   = {p2(0,0)/p2(2,0), p2(1,0)/p2(2,0)};



//    pIn2.x = min((float)(width-1), max(0.0f,pIn2.x));
//    pIn2.y = min((float)(height-1),max(0.0f,pIn2.y));

const cumat<3,1> p0withoutdepth = {(float)x,(float)y,(float)1};
const cumat<3,1> p2withoutdepth = KRKinv*p0withoutdepth;

/// KRK^(-1)[x y 1]';
float3 KRKinvpoint = make_float3(p2withoutdepth(0,0), p2withoutdepth(1,0), p2withoutdepth(2,0));

/// dIdx: Gradient [grad_Ix, grad_Iy]
float grad_Ix = tex2D(TexImg2,pIn2.x+1+0.5,pIn2.y+0.5) - tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5);
float grad_Iy = tex2D(TexImg2,pIn2.x+0.5,pIn2.y+1+0.5) - tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5);

float2 dIdx = make_float2(grad_Ix,grad_Iy);

/// grad_pi : Matrix 2x3
float3 grad_wrt_pi_d_u = make_float3(1/Pc2.z,0,-Pc2.x/(Pc2.z*Pc2.z));
float3 grad_wrt_pi_d_v = make_float3(0,1/Pc2.z,-Pc2.y/(Pc2.z*Pc2.z));

/// grad_pi dot KRKinvpoint
float2 grad_wrt_pi_times_KRKinv = make_float2(dot(grad_wrt_pi_d_u,KRKinvpoint) , dot(grad_wrt_pi_d_v,KRKinvpoint));

/// dIdd  = dIdx dot (grad_pi dot KRKinv)
float grad_I2_wrt_d_at_u0 = dot(grad_wrt_pi_times_KRKinv,dIdx);

if ( dividebyrange )
{
    grad_I2_wrt_d_at_u0  = grad_I2_wrt_d_at_u0/(dmax-dmin);
}


//    if ( x == 486 && y == 3 )
//    {
//        printf("Pc2 (x,y,z)  = %f %f %f\n",Pc2.x,Pc2.y,Pc2.z);
//        printf("(pIn2.x, pIn2.y) , gradient, u = %f %f %f %f\n", pIn2.x,pIn2.y,grad_I2_wrt_d_at_u0,du[y*stride+x]);
//    }


data_term[y*stride+x] = tex2D(TexImg2,pIn2.x+0.5,pIn2.y+0.5)- dI1[y*stride+x] + (du[y*stride+x]-du0[y*stride+x])*grad_I2_wrt_d_at_u0 ;
grad_wrt_d_at_d0[y*stride+x] = grad_I2_wrt_d_at_u0;
*/
