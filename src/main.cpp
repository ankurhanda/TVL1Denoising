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


extern "C" void launch_kernel_derivative_u(float* ux, float *uy, float* u, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height);
extern "C" void launch_kernel_dual_variable_p(float *px, float *py, float* ux, float *uy, float sigma, unsigned int stride, unsigned int mesh_width,
                                              unsigned int mesh_height);

extern "C" void launch_kernel_dual_variable_q(float *dq, float *u, float *g, float sigma, float lambda, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height);
extern "C" void launch_kernel_update_u(float *px, float *py, float *u, float* dq, unsigned int stride, unsigned int mesh_width, unsigned int mesh_height,
                                       float tau, float lambda);




int main( int /*argc*/, char* argv[] )
{

  cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
  pangolin::CreateGlutWindowAndBind("Main",512*2+150 ,512*2);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glewInit();


  View& d_panel = pangolin::CreatePanel("ui")
    .SetBounds(1.0, 0.0, 0, 150);

  CVD::Image<float> input_image;
  img_load(input_image, "../data/images/PockThesisImage.png");

  const unsigned width = input_image.size().x;
  const unsigned height = input_image.size().y;


  View& view_image0 = Display("image0").SetAspect(width*1.0/height*1.0);
  View& view_image1 = Display("image1").SetAspect(width*1.0/height*1.0);
  View& view_image2 = Display("image2").SetAspect(width*1.0/height*1.0);
  View& view_image3 = Display("image3").SetAspect(width*1.0/height*1.0);


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


  cout << "width = "<<width<<endl;
  cout << "height = "<<height<<endl;

  GlBufferCudaPtr pbo(GlPixelUnpackBuffer,  width*height*sizeof(float), cudaGraphicsMapFlagsNone, GL_STREAM_DRAW);
  GlTexture tex_show(width, height, GL_LUMINANCE);

  size_t imagePitchFloat;
  float *dq;
  float* px;
  float* py;
  float *ux, *uy, *u;
  float *g;





  cutilSafeCall(cudaMallocPitch(&(dq ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(px ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(py ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(ux ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(uy ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(u ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(g ), &(imagePitchFloat), width* sizeof (float), height));


  cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, input_image.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy2D(g, imagePitchFloat, input_image.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));


   cutilSafeCall(cudaMemset(px,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(py,0,sizeof(float)*width*height));

   cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));

   cutilSafeCall(cudaMemset(dq,0,sizeof(float)*width*height));

   unsigned int stride = (unsigned int)(imagePitchFloat/ sizeof(float));
   cout<< "stride = "<<stride <<endl;

  while(!pangolin::ShouldQuit())
  {

    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<bool> step("ui.Step", false, false);
    static Var<bool> continuous("ui.Run", false);
    static Var<double> lambda("ui.lambda", 0.0001 , 0, 4);



    if(HasResized())
      DisplayBase().ActivateScissorAndClear();

    glColor4f(1.0,1.0,1.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    static long unsigned int iterations = 0;

    if( Pushed(resetsq) )
    {
        cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, input_image.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy2D(g, imagePitchFloat, input_image.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));


        cutilSafeCall(cudaMemset(px,0,sizeof(float)*width*height));
        cutilSafeCall(cudaMemset(py,0,sizeof(float)*width*height));

        cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
        cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));

        cutilSafeCall(cudaMemset(dq,0,sizeof(float)*width*height));
    }

    if ( iterations % 100 == 0)
    {
//   for(int i = 0 ; i < 100; i ++){

         launch_kernel_derivative_u(ux,uy,u,stride, width , height);
         launch_kernel_dual_variable_p(px,py,ux,uy,0.5,stride,width,height);
         launch_kernel_dual_variable_q(dq,u,g,1/lambda,lambda, stride,width,height);
         launch_kernel_update_u(px,py,u,dq,stride,width,height,1/(lambda+4),lambda);

    }
    iterations++;


    // if ( iterations % 100 == 0)
    {
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,g,imagePitchFloat,pbo,tex_show);

        view_image1.Activate();
        DisplayFloatDeviceMem(&view_image1,u,imagePitchFloat,pbo,tex_show);

        view_image2.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image2,px,imagePitchFloat,pbo,tex_show);

        view_image3.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image3,py,imagePitchFloat,pbo,tex_show);
    }

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();

  }


  return 0;
}


