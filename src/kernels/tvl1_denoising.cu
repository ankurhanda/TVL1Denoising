/*
  */

#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


/////////////////////////////////////////////////////////////////////////////////
////! Simple kernel to modify vertex positions in sine wave pattern
////! @param data  data in global memory
/////////////////////////////////////////////////////////////////////////////////

__global__ void kernel(float *var, unsigned int stride, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output vertex
    var[y*stride+x] = var[y*stride+x];
}


__global__ void kernel_dualp(float *px, float *py, float *ux, float *uy, float sigma, unsigned int stride, unsigned int width, unsigned int height)
{


    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output vertex
    px[y*stride+x] = px[y*stride+x] + sigma* ux[y*stride+x];
    py[y*stride+x] = py[y*stride+x] + sigma* uy[y*stride+x];

    float pxval = px[y*stride+x];
    float pyval = py[y*stride+x];

    float reprojection = 0;
    reprojection   = sqrt(pxval*pxval + pyval*pyval);
    reprojection   = max(1,reprojection);

    px[y*stride+x] = px[y*stride+x]/reprojection;
    py[y*stride+x] = py[y*stride+x]/reprojection;


}


__global__ void kernel_dualq(float *dq, float *u, float* g, float sigma, float lambda, unsigned int stride, unsigned int width, unsigned int height)
{


    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output vertex
    dq[y*stride+x] = dq[y*stride+x] + sigma* lambda*(u[y*stride+x] - g[y*stride+x]);

    float reprojection = 0;
    reprojection   = fabs(dq[y*stride+x]);
    reprojection   = max(1,reprojection);

    dq[y*stride+x] = dq[y*stride+x]/reprojection;


}


__global__ void kernel_update_u(float *px, float *py, float *u, float* dq, unsigned int stride, unsigned int width, unsigned int height, float tau, float lambda)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float dxp = 0 , dyp = 0;

    if ( x >= 1 && x < width )  dxp = px[y*stride+x] - px[y*stride+(x-1)];

    if ( y >= 1 && y < height ) dyp = py[y*stride+x] - py[(y-1)*stride+x];

    float divp = dxp + dyp;

//    float u_prev = u[y*stride+x];

    u[y*stride+x] = (u[y*stride+x] + tau*(divp - lambda*dq[y*stride+x]));

//    u_[y*stride+x] = 2*u[y*stride+x] - u_prev;
   //  u_[y*stride+x] = u[y*stride+x];// - u_prev;


}


__global__ void kernel_derivative_u(float *ux, float *uy, float *u, unsigned int stride, unsigned int width, unsigned int height )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x + 1 < width ) //&& y < height )
    {
        ux[y*stride+x] = u[y*stride+(x+1)] - u[y*stride+x];
    }

    if ( y + 1 < height )
    {
        uy[y*stride+x] = u[(y+1)*stride+x] - u[y*stride+x];
    }

}

extern "C" void launch_kernel_derivative_u(float* ux, float *uy, float* u, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);


    kernel_derivative_u<<< grid, block>>>(ux, uy, u, stride, mesh_width, mesh_height);


    cutilCheckMsg("execution failed\n");


}


extern "C" void launch_kernel_update_u(float *px, float *py, float *u, float* dq, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height, float tau, float lambda)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel_update_u<<< grid, block>>>(px,py,u,dq, stride, mesh_width, mesh_height, tau, lambda);
    cutilCheckMsg("execution failed\n");
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_dual_variable_p(float *px, float *py, float* ux, float *uy, float sigma, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel_dualp<<< grid, block>>>(px,py,ux,uy,sigma, stride, mesh_width, mesh_height);
    cutilCheckMsg("execution failed\n");
}


extern "C" void launch_kernel_dual_variable_q(float *dq, float *u, float *g, float sigma, float lambda, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel_dualq<<< grid, block>>>(dq,u,g,sigma, lambda, stride, mesh_width, mesh_height);
    cutilCheckMsg("execution failed\n");
}



#endif // #ifndef _SIMPLEGL_KERNEL_H_
