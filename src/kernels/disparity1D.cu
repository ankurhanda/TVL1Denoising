#ifndef _DISPARITY1D_KERNEL_H_
#define _DISPARITY1D_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>

texture<float, 2, cudaReadModeElementType> my_tex;


const static cudaChannelFormatDesc chandesc_float1 =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define Nsize_hf_Width 1


__global__ void disparity_kernel_q(float* dq, float *du,
                                   float *du0,  float sigma_q, float lambda, float *dI1,
                                   unsigned int width, unsigned int height, unsigned int stride)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

//    float xinterp0 = max(0.0f,min(width*1.0,(float)x+ du0[y*stride+x]));
//    float xinterp1 = max(0.0f,min(width*1.0,(float)x+ du0[y*stride+x]+1));

    float xinterp0 = (float)x+ du0[y*stride+x];
    float xinterp1 = (float)x+ du0[y*stride+x]+1;

//    float xinterp1 = max(0.0f,min(width*1.0,(float)x+ du0[y*stride+x]+1));



    float I2_u0      = tex2D(my_tex,xinterp0+0.5,(float)y+0.5);
    float I1_val     = dI1[y*stride+x];
    float grad_I2_u0 = tex2D(my_tex,xinterp1+0.5,(float)y+0.5) - tex2D(my_tex,xinterp0+0.5,(float)y+0.5);

    float u  = du[y*stride+x];
    float u0 = du0[y*stride+x];

//    float data_term  = lambda*(I2_u0 + (u-u0)*grad_I2_u0 - I1_val);
//    float data_term  = lambda*((I2_u0-I1_val)*(I2_u0-I1_val) + (u-u0)*grad_I2_u0*(u-u0)*grad_I2_u0 + 2*(u-u0)*grad_I2_u0*(I2_u0-I1_val));

//    float mu2 = 0, mu1=0, corr = 0;
    int count = 0;
    float gradI2_sqr = 0, I2_u0_minus_I1 =0, I2_u0_minus_I1_times_grad=0;

    for (int i = -Nsize_hf_Width ; i <= Nsize_hf_Width ; i++)
    {
        for (int j = -Nsize_hf_Width ; j <= Nsize_hf_Width ; j++ )
        {

            if ( x+j < width && x-j >= 0 && y+i < height && y-i >= 0 )
            {
                   float xinterp0 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+(x+j)]));
                   float xinterp1 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+(x+j)]+1));

                   float grad_I2_u0 = tex2D(my_tex,xinterp1+0.5,(float)(y+i)+0.5) - tex2D(my_tex,xinterp0+0.5,(float)(y+i)+0.5);


                   gradI2_sqr += grad_I2_u0*grad_I2_u0;
                   I2_u0_minus_I1 += (tex2D(my_tex,xinterp0+0.5,y+i+0.5) - dI1[(y+i)*stride+x+j])*(tex2D(my_tex,xinterp0 + 0.5,y+i+0.5) - dI1[(y+i)*stride+x+j]);
                   I2_u0_minus_I1_times_grad += (tex2D(my_tex,xinterp0+0.5,y+i+0.5) - dI1[(y+i)*stride+(x+j)])*(grad_I2_u0);

//                   mu2 +=   tex2D(my_tex,xinterp0,y+i);
//                   mu1 +=   dI1[(y+i)*stride+(x+j)];
                   count++;
            }

        }

    }

    float data_term  = lambda*( I2_u0_minus_I1 + (u-u0)*(u-u0)*gradI2_sqr + (u-u0)*I2_u0_minus_I1_times_grad) ;

//    mu2 = mu2/(float)count;
//    mu1 = mu1/(float)count;

//    float sum_grad_I2_u0_times_I1 = 0;

//    for (int i = -3 ; i <= 3 ; i++)
//    {
//        for (int j = -3 ; j <= 3 ; j++ )
//        {
//            if ( x + j < width && x-j >=0 && y+i< height && y-i>=0 )
//            {
//                float xinterp0 = max(0.0f,min(width*1.0,(float)(x+j) + du0[(y+i)*stride+x+j]));
//                float xinterp1 = max(0.0f,min(width*1.0,(float)(x+j) + du0[(y+i)*stride+x+j]+1));

//                 corr += (tex2D(my_tex,xinterp0,y+i) - mu2)*(dI1[(y+i)*stride+(x+j)] - mu1);

//                 float grad_I2_u0 = tex2D(my_tex,xinterp1,(float)y) - tex2D(my_tex,xinterp0,(float)y);
//                 sum_grad_I2_u0_times_I1 += grad_I2_u0*(dI1[(y+i)*stride+(x+j)] - mu1);
//            }
//        }
//    }

//    float data_term = 1 - ( corr/count + (u-u0)*sum_grad_I2_u0_times_I1 / count );


    dq[y*stride+x] = dq[y*stride+x] + sigma_q*(data_term);

    // reprojection
    float reprojection_q = max(1.0f,fabs(dq[y*stride+x]));
    dq[y*stride+x] = dq[y*stride+x] / reprojection_q;
}

extern "C" void launch_disparity_kernel_q(float* dq, float *du,
                                          float *du0,  float sigma_q, float lambda,float *dI1,
                                          unsigned int width, unsigned int height, unsigned int stride)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    disparity_kernel_q<<<grid, block>>>(dq,du,du0,sigma_q,lambda,dI1,width,height,stride);


}

//launch_disparity_kernel_p (px,py,u,width,height,stride,sigma_p);

__global__ void disparity_kernel_p(float* dpx, float *dpy, float *du,
                                   unsigned int width, unsigned int height, unsigned int stride,
                                   float sigma_p)

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

    float pxval = dpx[y*stride+x] + sigma_p*(u_dx);
    float pyval = dpy[y*stride+x] + sigma_p*(u_dy);

    // reprojection
    float reprojection_p   = max(1.0f,sqrt(pxval*pxval + pyval*pyval));

    dpx[y*stride+x] = pxval / reprojection_p;
    dpy[y*stride+x] = pyval / reprojection_p;
}

extern "C" void launch_disparity_kernel_p(float* dpx, float *dpy, float *du,
                                          unsigned int width, unsigned int height, unsigned int stride,
                                          float sigma_p)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    disparity_kernel_p<<<grid, block>>>(dpx, dpy, du, width,height,stride,sigma_p);

}

__global__ void disparity_kernel_u(float* dpx, float* dpy,float *du, float *du0, float *dq, float *dI1,
                                   float sigma_u, float lambda, unsigned int width,
                                   unsigned height, unsigned int stride)
{
    float dxp = 0 , dyp = 0;

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x >= 1 && x < width )  dxp = dpx[y*stride+x] - dpx[y*stride+(x-1)];

    if ( y >= 1 && y < height ) dyp = dpy[y*stride+x] - dpy[(y-1)*stride+x];

    float div_p = dxp + dyp;

    float xinterp0 = max(0.0f,min(width,(float)x+ du0[y*stride+x]));
    float xinterp1 = max(0.0f,min(width,(float)x+ du0[y*stride+x]+1));

//    float xinterp0 = (float)x+ du0[y*stride+x];
//    float xinterp1 = (float)x+ du0[y*stride+x]+1;

    float grad_I2_u0 = tex2D(my_tex,xinterp1+0.5,(float)y+0.5) - tex2D(my_tex,xinterp0+0.5,(float)y+0.5);

    float I1_val = dI1[y*stride+x];

//    float diff_term = lambda*dq[y*stride+x]*grad_I2_u0 - div_p;

//    float diff_term = lambda*(2*(du[y*stride+x]-du0[y*stride+x])*(grad_I2_u0)*grad_I2_u0 + 2*grad_I2_u0*(tex2D(my_tex,xinterp0,(float)y) - I1_val)) - div_p;


//    float mu2=0,mu1=0;
    int count = 0;
//    for (int i = -3 ; i <= 3 ; i++)
//    {
//        for (int j = -3 ; j <= 3 ; j++ )
//        {

//            if ( x+j < width && x-j >= 0 && y+i < height && y-i >= 0 )
//            {
//                   float xinterp0 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+(x+j)]));
//                   mu2 += tex2D(my_tex,xinterp0,y+i);
//                   mu1 += dI1[(y+i)*stride+(x+j)];
//                   count++;
//            }

//        }

//    }

//    mu2 = mu2/(float)count;
//    mu1 = mu1/(float)count;

//    float sum_grad_I2_u0_times_I1 = 0;

//    for (int i = -3 ; i <= 3 ; i++)
//    {
//        for (int j = -3 ; j <= 3 ; j++ )
//        {
//            if ( x + j < width && x-j >=0 && y+i< height && y-i>=0 )
//            {
//                float xinterp0 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+x+j]));
//                float xinterp1 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+x+j]+1));
//                float grad_I2_u0 = tex2D(my_tex,xinterp1,(float)y) - tex2D(my_tex,xinterp0,(float)y);

//                sum_grad_I2_u0_times_I1 += grad_I2_u0*(dI1[(y+i)*stride+(x+j)] - mu1);
//            }
//        }
//    }

//    float diff_term = sum_grad_I2_u0_times_I1 / count;


    float gradI2_sqr = 0, I2_u0_minus_I1_times_grad=0;

    for (int i = -Nsize_hf_Width ; i <= Nsize_hf_Width ; i++)
    {
        for (int j = -Nsize_hf_Width ; j <= Nsize_hf_Width ; j++ )
        {

            if ( x+j < width && x-j >= 0 && y+i < height && y-i >= 0 )
            {
                   float xinterp0 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+(x+j)]));
                   float xinterp1 = max(0.0f,min(width*1.0,(float)(x+j)+ du0[(y+i)*stride+(x+j)]+1));

                   float grad_I2_u0 = tex2D(my_tex,xinterp1+0.5,(float)(y+i)+0.5) - tex2D(my_tex,xinterp0+0.5,(float)(y+i)+0.5);

                       gradI2_sqr += grad_I2_u0*grad_I2_u0;
                   I2_u0_minus_I1_times_grad += (tex2D(my_tex,xinterp0+0.5,y+i+0.5) - dI1[(y+i)*stride+(x+j)])*(grad_I2_u0);

//                   mu2 +=   tex2D(my_tex,xinterp0,y+i);
//                   mu1 +=   dI1[(y+i)*stride+(x+j)];
                   count++;
            }

        }

    }

    float diff_term = lambda*(2*(du[y*stride+x]-du0[y*stride+x])*gradI2_sqr+ 2*I2_u0_minus_I1_times_grad) - div_p;


    du[y*stride+x]  = du[y*stride+x] - sigma_u*(diff_term);
}


extern "C" void launch_disparity_kernel_u(float* dpx, float* dpy,float *du, float *du0,float *dq, float *dI1,
                                          float sigma_u, float lambda, unsigned int width,
                                          unsigned int height, unsigned int stride)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    disparity_kernel_u<<<grid,block>>>(dpx,dpy,du,du0,dq,dI1,sigma_u,lambda,width,height,stride);

}

extern "C" void launch_disparity_kernel(float* d_I2, /*float* dq, float *u,
                                        float* u0,   float* px, float *py,*/
                                        unsigned int width, unsigned int height, unsigned int imgStride
                                        /*, float sigma_p, float sigma_q, float lambda*/)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);

    cudaBindTexture2D(0,my_tex,d_I2,chandesc_float1,width, height,imgStride*sizeof(float));
    my_tex.addressMode[0] = cudaAddressModeClamp;
    my_tex.addressMode[1] = cudaAddressModeClamp;
    my_tex.filterMode = cudaFilterModeLinear;
    my_tex.normalized = false;    // access with normalized texture coordinates


//    kernel_disparity_estimation<<< grid, block>>>(u, u0, p, q, I1,sigma_q, sigma_p, sigma_u, lambda);
}

__global__ void disparity_kernel_copy_u0_to_u(float* du, float *du0,
                                   unsigned int width, unsigned int height, unsigned int stride)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    du0[y*stride+x] = du[y*stride+x];
}

extern "C" void  launch_disparity_kernel_copy_u0_to_u(float *du,float *du0, unsigned int width, unsigned int height, unsigned int stride)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    disparity_kernel_copy_u0_to_u<<<grid,block>>>(du, du0, width,height,stride);
}

__global__ void disparity_kernel_I2warped(float *dI2warped,float *du, unsigned int width, unsigned int height, unsigned int stride)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float xinterp0 = max(0,min(width*1.0,(float)x+ du[y*stride+x]));
//    float xinterp1 = max(0,min(width*1.0,(float)x+ du0[y*stride+x]+1));


    dI2warped[y*stride+x] = tex2D(my_tex,xinterp0+0.5,y+0.5);
}

extern "C" void launch_disparity_kernel_I2warped(float *dI2warped,float *du, unsigned int width, unsigned int height, unsigned int stride)
{
    dim3 block(8,8,1);
    dim3 grid(width / block.x, height / block.y, 1);
    disparity_kernel_I2warped<<<grid,block>>>(dI2warped, du, width,height,stride);
}




#endif // #ifndef _DISPARITY1D_KERNEL_H_
