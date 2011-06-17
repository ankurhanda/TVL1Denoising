#include "utils.h"

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
