/**
 * @author  Ankur Handa
 * Copyright (C) 2011  Ankur Handa
 *                     Imperial College London
 **/

#include <GL/glew.h>

#include <boost/thread.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <cvd/videosource.h>
#include <TooN/sl.h>
#include <TooN/so3.h>
#include <TooN/se3.h>
#include <TooN/LU.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include "utils.h"

#define HAVE_TOON

using namespace pangolin;
using namespace std;
using namespace CVD;
using namespace TooN;




int main( int /*argc*/, char* argv[] )
{

  cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
  pangolin::CreateGlutWindowAndBind("Main",1024,768);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glewInit();


  View& d_panel = pangolin::CreatePanel("ui")
    .SetBounds(1.0, 0.0, 0, 150);

  View& view_image0 = Display("image0").SetAspect(640.0/480.0);
  View& view_image1 = Display("image1").SetAspect(640.0/480.0);
  View& view_image2 = Display("image2").SetAspect(640.0/480.0);
  View& view_image3 = Display("image3").SetAspect(640.0/480.0);


//  View& d_imgs = pangolin::Display("images")
//    .SetBounds(1.0, 0.0, 150, 1.0, false)
//    .SetLayout(LayoutEqual)
//    .AddDisplay(*d_img[0])
//    .AddDisplay(*d_img[1])
//    .AddDisplay(*d_img[2])
//    .AddDisplay(*d_img[3]);

  View& d_imgs = pangolin::Display("images")
                     //.SetBounds(1.0, 0.8, 150, 1.0, false)
                     //first distance from bottom left of opengl window i.e. 0.7 is 70%
                     //co-ordinate from bottom left of screen from
                     //0.0 to 1.0 for top, bottom, left, right.
                     .SetBounds(1.0, 0.0, 150/*cus this is width in pixels of our panel*/, 1.0, true)
                     .SetLayout(LayoutEqual)
                     .AddDisplay(view_image0)
                     .AddDisplay(view_image1)
                     .AddDisplay(view_image2)
                     .AddDisplay(view_image3)
                     ;


//  View& d_3d = pangolin::Display("3d")
//    .SetBounds(1.0, 0.0, 150, 1.0, -640/480.0)
//    .SetHandler(new Handler3D(s_cam));


  CVD::Image<float> input_image;
  img_load(input_image, "../data/images/lena_640x480.png");


  const unsigned width = input_image.size().x;
  const unsigned height = input_image.size().y;

  cout << "width = "<<width<<endl;
  cout << "height = "<<height<<endl;

  GlBufferCudaPtr pbo(GlPixelUnpackBuffer,  width*height*sizeof(float), cudaGraphicsMapFlagsNone, GL_STREAM_DRAW);
  GlTexture tex_show(width, height, GL_LUMINANCE);


//  GlBufferCudaPtr greypbo( GlPixelUnpackBuffer, width*height*sizeof(float), cudaGraphicsMapFlagsNone, GL_STREAM_DRAW );

  size_t imagePitchFloat;
//  float *dq;
//  float* px;
//  float* py;
//  float *ux, *uy, *u;
//  float *g;

    float* u;



//  cutilSafeCall(cudaMallocPitch(&(dq ), &(imagePitchFloat), width* sizeof (float), height));

//  cutilSafeCall(cudaMallocPitch(&(px ), &(imagePitchFloat), width* sizeof (float), height));
//  cutilSafeCall(cudaMallocPitch(&(py ), &(imagePitchFloat), width* sizeof (float), height));

//  cutilSafeCall(cudaMallocPitch(&(ux ), &(imagePitchFloat), width* sizeof (float), height));
//  cutilSafeCall(cudaMallocPitch(&(uy ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(u ), &(imagePitchFloat), width* sizeof (float), height));

//  cutilSafeCall(cudaMallocPitch(&(g ), &(imagePitchFloat), width* sizeof (float), height));


  cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, input_image.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));


//   cutilSafeCall(cudaMemset(px,0,sizeof(float)*width*height));
//   cutilSafeCall(cudaMemset(py,0,sizeof(float)*width*height));

//   cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
//   cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));

//   cutilSafeCall(cudaMemset(dq,0,sizeof(float)*width*height));



//  for(int frame_num=0; !pangolin::ShouldQuit();)
  while(!pangolin::ShouldQuit())
  {
//    const bool run = continuous || Pushed(step);
    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<bool> step("ui.Step", false, false);
    static Var<bool> continuous("ui.Run", false);
    static Var<double> lambda("ui.lambda", 0.01, 0, 50);
    static Var<double> tracking_err("ui.TErr");


    if(HasResized())
      DisplayBase().ActivateScissorAndClear();

    // Show Images
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(1.0,1.0,1.0,1.0);
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    {
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,u,imagePitchFloat,pbo,tex_show);

//        view_image1.ActivateAndScissor();
//        DisplayFloatDeviceMem(&view_image1,u,imagePitchFloat,pbo,tex_show);

//        view_image2.ActivateAndScissor();
//        DisplayFloatDeviceMem(&view_image2,u,imagePitchFloat,pbo,tex_show);

//        view_image3.ActivateAndScissor();
//        DisplayFloatDeviceMem(&view_image3,u,imagePitchFloat,pbo,tex_show);
    }


//        CudaScopedMappedPtr cu(pbo);
//        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float), u, imagePitchFloat, tex_show.width * sizeof (float), tex_show.height, cudaMemcpyDeviceToDevice));
//        CopyPboToTex(pbo,tex_show, GL_LUMINANCE32F_ARB, GL_FLOAT );
//        tex_show.RenderToViewportFlipY();


//    d_img[1]->ActivateAndScissor();
//    CopyPboToTex(live->pbo_im,tex_show, GL_RGBA, GL_FLOAT );
//    tex_show.RenderToViewportFlipY();

//    d_img[2]->ActivateAndScissor();
//    CopyPboToTex(pbo_debug,tex_show, GL_RGBA, GL_FLOAT );
//    tex_show.RenderToViewportFlipY();

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();
//    boost::this_thread::sleep(boost::posix_time::milliseconds(10));
  }


  return 0;
}


