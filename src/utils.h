#ifndef UTILS_H_
#define UTILS_H_

#include <limits>
#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include <cvd/image.h>
#include <cvd/image_io.h>
#include <cvd/image_ref.h>


using namespace std;
using namespace pangolin;



struct ScopedCuTimer{

    unsigned int timer;
    std::string name;
    bool doThreadSync;
    float last_time;


    inline float GetDeltaTime_s()
    {
        const float delta = cutGetTimerValue(timer) / 1000.0f;
        cutilCheckError(cutStopTimer(timer));
        cutilCheckError(cutResetTimer(timer));
        cutilCheckError(cutStartTimer(timer));
        return delta;
    }

    inline ScopedCuTimer(std::string name, bool doThreadSync=true):name(name),doThreadSync(doThreadSync){
        timer=0;
        last_time = 0;
        cutilCheckError( cutCreateTimer( &timer));
        if(doThreadSync) cudaThreadSynchronize();
        cutilCheckError(cutStartTimer(timer));
    };
    inline ~ScopedCuTimer(){
        if(doThreadSync) cudaThreadSynchronize();
        cutilCheckError(cutStopTimer(timer));
        std::cout<<  name << "::" << (float)cutGetTimerValue(timer)<< "ms" <<std::endl;
        cutilCheckError(cutDeleteTimer(timer));
    }
};


inline void DisplayFloatPBO(View* view, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    view->ActivateAndScissor();
    CopyPboToTex(pbo,tex_show, GL_LUMINANCE, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloat4PBO(View* view, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    view->ActivateAndScissor();
    CopyPboToTex(pbo,tex_show, GL_RGBA, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloat4PBO(GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    CopyPboToTex(pbo,tex_show, GL_RGBA, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloatDeviceMem(View* view, float* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {
        CudaScopedMappedPtr cu(pbo);
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float), tex_show.height, cudaMemcpyDeviceToDevice));
    }
    DisplayFloatPBO(view, pbo, tex_show);
}

inline void DisplayFloat4DeviceMem(View* view, float4* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {
        CudaScopedMappedPtr cu(pbo);
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float4), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float4), tex_show.height, cudaMemcpyDeviceToDevice));
    }


    DisplayFloat4PBO(view, pbo, tex_show);
}

inline void DisplayFloat4DeviceMem(float4* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {
        CudaScopedMappedPtr cu(pbo);
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float4), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float4), tex_show.height, cudaMemcpyDeviceToDevice));
    }
    DisplayFloat4PBO(pbo, tex_show);
}


void setImageData(float * imageArray, int width, int height){
    for(int i = 0 ; i < width*height;i++) {
        imageArray[i] = (float)rand()/RAND_MAX;
    }
}



#endif
