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
//#include "./kernels/tvl1_denoising.cu"

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



extern "C" void launch_disparity_kernel_q(float* dq, float *du,
                                          float *du0,  float sigma_q, float lambda,float *dI1,
                                          unsigned int width, unsigned int height, unsigned int stride);

extern "C" void launch_disparity_kernel_p(float* dpx, float *dpy, float *du,
                                          unsigned int width, unsigned int height, unsigned int stride,
                                          float sigma_p);

extern "C" void launch_disparity_kernel_u(float* dpx, float* dpy,float *du, float *du0,float *dq,
                                          float sigma_u, float lambda, unsigned int width,
                                          unsigned int height, unsigned int stride);


extern "C" void launch_disparity_kernel(float* d_I2, unsigned int width, unsigned int height, unsigned int imgStride);

extern "C" void launch_disparity_kernel_copy_u0_to_u(float *du,float *du0, unsigned int width, unsigned int height, unsigned int stride);
extern "C" void launch_disparity_kernel_I2warped(float *dI2warped,float *du, unsigned int width, unsigned int height, unsigned int stride);



//extern "C" texture<float, 2, cudaReadModeElementType> my_tex;


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


  CVD::Image<float> Image1;
//  img_load(Image1, "../data/images/PockThesisImage.png");
  img_load(Image1, "../data/Baby2/view1.png");

  CVD::Image<float> Image2;
  img_load(Image2, "../data/Baby2/view2.png");

//  const unsigned width = input_image.size().x;
//  const unsigned height = input_image.size().y;

  const unsigned width = Image1.size().x;
  const unsigned height = Image1.size().y;


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
  float *ux, *uy, *u, *u0;
  float *g;
  float *dI1;
  float *dI2;
  float *dI2warped;
  float *udisp, *pxdisp;

  float* h_udisp = new float[width*height];
  float* h_pxdisp = new float[width*height];

  float* u0data = new float [width*height];


  cutilSafeCall(cudaMallocPitch(&(dq ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(px ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(py ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(ux ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(uy ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(u ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(g ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(dI1 ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(dI2 ), &(imagePitchFloat), width* sizeof (float), height));



  cutilSafeCall(cudaMallocPitch(&(udisp ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(pxdisp ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(u0 ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(dI2warped ), &(imagePitchFloat), width* sizeof (float), height));



  cutilSafeCall(cudaMemcpy2D(g, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMemcpy2D(dI1, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy2D(dI2, imagePitchFloat, Image2.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMemcpy2D(dI2warped, imagePitchFloat, Image2.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));




  cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

   cutilSafeCall(cudaMemset(px,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(py,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));
   cutilSafeCall(cudaMemset(dq,0,sizeof(float)*width*height));

   cutilSafeCall(cudaMemset(dI2warped,0,sizeof(float)*width*height));

   unsigned int stride = (unsigned int)(imagePitchFloat/ sizeof(float));
   cout<< "stride = "<<stride <<endl;


//   // allocate array and copy image data
//   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//   cudaArray* cu_array;
//   cutilSafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height ));
//   cutilSafeCall( cudaMemcpyToArray( cu_array, 0, 0, h_data, size, cudaMemcpyHostToDevice));

//   // set texture parameters
//   tex.addressMode[0] = cudaAddressModeWrap;
//   tex.addressMode[1] = cudaAddressModeWrap;
//   tex.filterMode = cudaFilterModeLinear;
//   tex.normalized = true;    // access with normalized texture coordinates

//   // Bind the array to the texture
//   cutilSafeCall( cudaBindTextureToArray( tex, cu_array, channelDesc));

   int when2show = 0;
   int warps = 0;


  while(!pangolin::ShouldQuit())
  {

    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<bool> step("ui.Step", false, false);
    static Var<bool> continuous("ui.Run", false);
    static Var<double> lambda("ui.lambda", 0.02 , 0, 100);

    static Var<double> sigma_p("ui.sigma_p", 0.5 , 0, 4);
    static Var<double> sigma_q("ui.sigma_q", 0.02 , 0, 4);
    static Var<double> sigma_u("ui.sigma_u", 0.05 , 0, 4);

    static Var<int> max_iterations("ui.max_iterations", 300 , 1, 4000);
    static Var<int> max_warps("ui.max_warps", 20 , 1, 400);

    static Var<int> u0initval("ui.u0initval", -8 , -10, 10);




    if(HasResized())
      DisplayBase().ActivateScissorAndClear();

    glColor4f(1.0,1.0,1.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    static long unsigned int iterations = 0;

    if ( warps == 0 && iterations == 0)
    {
        for (int i = 0 ; i < width*height; i++)
        {
            u0data[i] = (float)u0initval;
        }
        cutilSafeCall(cudaMemcpy2D(u0,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy2D(u,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

    }

    if( Pushed(resetsq))
    {
//        cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy2D(g,   imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy2D(dI2, imagePitchFloat, Image2.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

        warps = 0 ;
        iterations = 0;
        if ( warps == 0 )
        {
            for (int i = 0 ; i < width*height; i++)
            {
                u0data[i] = (float)u0initval;
            }

        }

        cutilSafeCall(cudaMemcpy2D(u0,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
        cutilSafeCall(cudaMemcpy2D(u,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));


        cutilSafeCall(cudaMemset(px,0,sizeof(float)*width*height));
        cutilSafeCall(cudaMemset(py,0,sizeof(float)*width*height));

        cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
        cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));

        cutilSafeCall(cudaMemset(dq,0,sizeof(float)*width*height));
        cutilSafeCall(cudaMemset(dI2warped,0,sizeof(float)*width*height));

    }

    if ( iterations > max_iterations)
    {
        iterations = 0;

        launch_disparity_kernel_copy_u0_to_u(u,u0,width, height,stride);

        warps++;
    }

//    if ( iterations % 100 == 0)
    if ( warps <= max_warps &&  when2show % 10 == 0 )
    {

//   for(int i = 0 ; i < 100; i ++){
//         launch_kernel_derivative_u(ux,uy,u,stride, width , height);
//         launch_kernel_dual_variable_p(px,py,ux,uy,0.5,stride,width,height);
//         launch_disparity_kernel_p (px,py,u,width,height,stride,sigma_p);
//         launch_kernel_dual_variable_q(dq,u,g,1/lambda,lambda, stride,width,height);
//         launch_kernel_update_u(px,py,u,dq,stride,width,height,1/(lambda+4),lambda);
//        launch_disparity_kernel_u (px,py,u,u0,dq,1/(lambda+4),lambda,width,height,stride);

//        for (int  j = 0; j < max_warps ; j++)
//            for ( int i = 0 ; i < max_iterations ; i++)

//            {
                launch_disparity_kernel   (dI2,width,height,stride);
                launch_disparity_kernel_q (dq,u,u0,sigma_q,lambda,g,width,height,stride); // g = I1;
                launch_disparity_kernel_p (px,py,u,width,height,stride,sigma_p);
                launch_disparity_kernel_u (px,py,u,u0,dq,sigma_u,lambda,width,height,stride);
                launch_disparity_kernel_I2warped(dI2warped,u,width,height,stride);

//                launch_disparity_kernel_copyu_display(udisp,u,width,height,stride);
//                launch_disparity_kernel_copypx_display(pxdisp,px,width,height,stride);
//            }
            iterations++;

//            cudaMemcpy2D(matrix, width * sizeof(float), dev_matrix, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);

            cudaMemcpy2D(h_udisp,width*sizeof(float),u,imagePitchFloat,width*sizeof(float),height,cudaMemcpyDeviceToHost);
            cudaMemcpy2D(h_pxdisp,width*sizeof(float),px,imagePitchFloat,width*sizeof(float),height,cudaMemcpyDeviceToHost);

//            cutilSafeCall(cudaMemcpy2D(h_udisp ,  imagePitchFloat, u, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
//            cutilSafeCall(cudaMemcpy2D(h_pxdisp,  imagePitchFloat, px, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyDeviceToHost));


            float max_u=-9999.0f;
            float min_u = 9999.0f;
            float max_p = max_u;
            float min_p = min_u;

            for (int i = 0 ; i < width*height; i++)
            {
                if ( h_udisp[i] > max_u)
                {
                    max_u = h_udisp[i];
                }
                if ( h_udisp[i] < min_u)
                {
                    min_u = h_udisp[i];
                }

                if ( h_pxdisp[i] > max_p)
                {
                    max_p = h_pxdisp[i];
                }
                if ( h_pxdisp[i] < min_p)
                {
                    min_p = h_pxdisp[i];
                }
            }

            for (int i = 0 ; i < width*height; i++)
            {
                h_udisp[i] = (h_udisp[i]-min_u)/(max_u-min_u);
                h_pxdisp[i] = (h_pxdisp[i]-min_p)/(max_p-min_p);

            }

            cutilSafeCall(cudaMemcpy2D(udisp ,  imagePitchFloat, h_udisp, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy2D(pxdisp,  imagePitchFloat, h_pxdisp, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

    }
    when2show++;

    cout << "iterations = " << iterations << endl;


    // if ( iterations % 100 == 0)
    {
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,g,imagePitchFloat,pbo,tex_show);

        view_image1.Activate();
        DisplayFloatDeviceMem(&view_image1,udisp,imagePitchFloat,pbo,tex_show);

        view_image2.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image2,dI2warped,imagePitchFloat,pbo,tex_show);

        view_image3.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image3,py,imagePitchFloat,pbo,tex_show);
    }

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();

  }


  return 0;
}


