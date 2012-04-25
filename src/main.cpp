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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#include "../imageutilities/src/iucore.h"

#include "utils.h"
#include "./depthest/TVL1DepthEstimation.h"
#include "kernels/Strumdepthestimation.h"

#define HAVE_TOON

using namespace pangolin;
using namespace std;
using namespace CVD;
using namespace TooN;

int main( int /*argc*/, char* argv[] )
{

  cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
  pangolin::CreateGlutWindowAndBind("Main",512*2+150 ,512*2);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glewInit();


  View& d_panel = pangolin::CreatePanel("ui")
    .SetBounds(1.0, 0.0, 0, Attach::Pix(150));

  TVL1DepthEstimation *TGVDenoising = new TVL1DepthEstimation("../data/lena_sigma25.png");

  int width  = TGVDenoising->getImgWidth();
  int height = TGVDenoising->getImgHeight();

  cout << "Width = "<< width << ", Height = " << height << endl;

  View& view_image0 = Display("image0").SetAspect(width*1.0/height*1.0);
  View& view_image1 = Display("image1").SetAspect(width*1.0/height*1.0);
  View& view_image2 = Display("image2").SetAspect(width*1.0/height*1.0);
  View& view_image3 = Display("image3").SetAspect(width*1.0/height*1.0);


  View& d_imgs = pangolin::Display("images")
                     .SetBounds(1.0, 0.0, Attach::Pix(150)/*cus this is width in pixels of our panel*/, 1.0, true)
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

  iu::ImageGpu_32f_C1 u_disp(IuSize(width,height));


  while(!pangolin::ShouldQuit())
  {

    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<float> lambda("ui.lambda", 0.011, 0, 5);
    //float lambda = lambda_ / (mvsimagesfilename.size());

    static Var<float> sigma_p("ui.sigma_p", 0.5 , 0, 4);
    static Var<float> sigma_q("ui.sigma_q", 0.02 , 0, 4);
    static Var<float> tau("ui.tau", 0.05 , 0, 4);

    static Var<int> max_iterations("ui.max_iterations", 300 , 1, 4000);
    static Var<int> max_warps("ui.max_warps", 20 , 0, 400);

     static Var<float> u0initval("ui.u0initval", 0.5 , 0, 1);

    static Var<float> dmin("ui.dmin", 0.01 , 0, 2);
    static Var<float> dmax("ui.dmax", 1 , 0, 4);
    static Var<float> epsilon("ui.epsilon", 1E-4 , 1E-3, 1E-2);



    if(HasResized())
      DisplayBase().ActivateScissorAndClear();

    glColor4f(1.0,1.0,1.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    static long unsigned int iterations = 0;

    if( Pushed(resetsq) )
    {
        TGVDenoising->InitialiseVariables(u0initval);
    }

    TGVDenoising->updateDualP(sigma_p, lambda);
    TGVDenoising->updateDualQ(sigma_p, 2*lambda);
    TGVDenoising->updatePrimalV(tau, 2*lambda);
    TGVDenoising->updatePrimalU(tau);

//    if (warps == 0 && iterations == 0 )
//    {
//        cout <<"Computing the gradients" << endl;
//        Stereo2D->initMVSdataOnly();
//        for(int img_no = 0 ; img_no < mvsimagesfilename.size() ; img_no++)
//        {
//            R_lr_ = R_lr_vector.at(img_no);
//            t_lr_ = t_lr_vector.at(img_no);
//            Stereo2D->computeImageGradient_wrt_depth(fl,
//                                                     pp,
//                                                     R_lr_,
//                                                     t_lr_,
//                                                     compute_disparity,
//                                                     dmin,
//                                                     dmax,
//                                                     img_no);
//        }

//        Stereo2D->sortCriticalPoints();
//        cout << "Has computed the gradients" << endl;
//    }

//    if ( iterations > max_iterations)
//    {
//        iterations = 0;
//        cout << "Warping going on!"<<endl;
//        Stereo2D->doOneWarp();
//        Stereo2D->initMVSdataOnly();
//        for(int img_no = 0 ; img_no < mvsimagesfilename.size() ; img_no++)
//        {
//            R_lr_ = R_lr_vector.at(img_no);
//            t_lr_ = t_lr_vector.at(img_no);
//            Stereo2D->computeImageGradient_wrt_depth(fl,
//                                                     pp,
//                                                     R_lr_,
//                                                     t_lr_,
//                                                     compute_disparity,
//                                                     dmin,
//                                                     dmax,
//                                                     img_no);
//        }
//        Stereo2D->sortCriticalPoints();
//        warps++;
//    }

//    if (warps <= max_warps &&  when2show % 1  == 0 )
//    {

//         Stereo2D->updatedualReg(lambda,
//                               tau,
//                               sigma_q,
//                               sigma_p,
//                               epsilon);


//         Stereo2D->updatePrimalData(lambda,
//                                tau,
//                                sigma_q,
//                                sigma_p);

//         Stereo2D->updateWarpedImage(fl,
//                                    pp,
//                                    R_lr_,
//                                    t_lr_,
//                                    compute_disparity);

//         iterations++;



//         cout << "Is everything going okay?" << endl;
//         iu::copy(Stereo2D->d_u,&h_udisp);
////         cudaMemcpy(h_udisp.data(),Stereo2D->d_u->data(),width*height*sizeof(float),cudaMemcpyDeviceToHost);


//         cout << "Yes, it is!" << endl;

//            float max_u =-9999.0f;
//            float min_u = 9999.0f;
//            float max_p = max_u;
//            float min_p = min_u;
//            int max_row, min_row;
//            int max_col, min_col;

//            for (int i = 0 ; i < height; i++)
//            {
//                for(int j = 0 ; j < width ; j++ )
//                {
//                    if ( h_udisp.getPixel(j,i) > max_u)
//                    {
//                        max_u = h_udisp.getPixel(j,i);
//                        max_col = j;
//                        max_row = i;
//                    }
//                    if ( h_udisp.getPixel(j,i) < min_u)
//                    {
//                        min_u = h_udisp.getPixel(j,i);
//                        min_col = j;
//                        min_row = i;
//                    }

//                }
//            }

//            //iu::copy(h_udisp,)
//            float *h_udisp_ptr = h_udisp.data();
//            for(int i = 0 ; i < height; i++)
//            {
//                for(int j = 0 ; j < width; j++)
//                {
//                    float val = *(h_udisp_ptr+(i*width+j));
////                    cout << "before = " << val << endl;
//                    *(h_udisp_ptr+(i*width+j)) = (val - min_u)/(max_u - min_u);
////                    cout << " h_udisp.getPixel(j,i)" << h_udisp.getPixel(j,i) << endl;
//                }
//            }

//            iu::copy(&h_udisp,&u_disp);
//            cout << "max_u =" << max_u << ", min_u= "<< min_u << endl;
////            cout << "max_p =" << max_p << ", min_p= "<< min_p << endl;
//    }
//    when2show++;

//    cout << "iterations = " << iterations << endl;


//     if ( iterations % 100 == 0)
    {
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,TGVDenoising->d_ref_image->data(),TGVDenoising->d_ref_image->pitch(),pbo,tex_show);

        view_image1.Activate();
        DisplayFloatDeviceMem(&view_image1,TGVDenoising->d_u->data(),TGVDenoising->d_u->pitch(),pbo,tex_show);

        view_image2.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image2,TGVDenoising->d_px->data(),TGVDenoising->d_px->pitch(),pbo,tex_show);

        view_image3.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image3,TGVDenoising->d_px->data(),TGVDenoising->d_px->pitch(),pbo,tex_show);
    }

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();

  }


  return 0;
}


