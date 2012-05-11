#ifndef _CPP_CUDA_VISION_KERNEL_DERIVATIVES_CUH_
#define _CPP_CUDA_VISION_KERNEL_DERIVATIVES_CUH_

#include <cutil_inline.h>
#include <cutil_math.h>

#define getImageIdxStrided(x,y,stride)  ((y)*(stride) + (x))



inline __device__  float dxm(float * __restrict__ p,const uint x,const uint y, const int2 d_imageSize, const size_t stridef1) {

    if (x == 0) {
        return p[getImageIdxStrided(x,y,stridef1)];
    } else if (x ==d_imageSize.x - 1) {
        return - p[getImageIdxStrided(x-1,y,stridef1)];
    } else {
        return  p[getImageIdxStrided(x,y,stridef1)] -  p[getImageIdxStrided(x-1,y,stridef1)];
    }
}

inline __device__   float dym(float * __restrict__ p,const uint x,const uint y,  const int2 d_imageSize, const size_t stridef1) {

    if (y == 0) {
        return p[getImageIdxStrided(x,y,stridef1)];
    } else if (y == d_imageSize.y - 1) {
        return -p[getImageIdxStrided(x,y-1,stridef1)];
    } else {
        return p[getImageIdxStrided(x,y,stridef1)] - p[getImageIdxStrided(x,y-1,stridef1)];
    }
}

inline __device__  float dxp(float * __restrict__ p,const uint x,const uint y, const int2 d_imageSize, size_t stridef1) {

    //(14) page 4
    if (x == d_imageSize.x - 1) {
        return 0;
    } else {
        return  p[getImageIdxStrided(x+1,y,stridef1)] - p[getImageIdxStrided(x,y,stridef1)];
    }
}

inline __device__  float dyp(float * __restrict__ p,const uint x,const uint y, const int2 d_imageSize, const size_t stridef1) {

    if (y == d_imageSize.y - 1) {
        return 0;
    } else {
        return p[getImageIdxStrided(x,y+1,stridef1)] - p[getImageIdxStrided(x,y,stridef1)];
    }
}






#endif
