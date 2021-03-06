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
#include "../imageutilities/src/iumath.h"
#include "../imageutilities/src/iucutil.h"

#include "utils.h"
#include "./depthest/TVL1DepthEstimation.h"

//#include "./kernels/tvl1_denoising.cu"

#include "kernels/Strumdepthestimation.h"



#define HAVE_TOON

using namespace pangolin;
using namespace std;
using namespace CVD;
using namespace TooN;


//extern "C" texture<float, 2, cudaReadModeElementType> my_tex;

struct GPUMem
{
 GPUMem() :
   free_mbyte(0),
   total_mbyte(0),
   used_mbyte(0)
 {}

 size_t free_mbyte;
 size_t total_mbyte;
 size_t used_mbyte;

 void getGPUMem()
 {
   cudaMemGetInfo(&free_mbyte, &total_mbyte);

   free_mbyte /= (1024*1024);
   total_mbyte /= (1024*1024);
   used_mbyte = total_mbyte - free_mbyte;
 }
};



TooN::SE3<> computeTpov_cam(int ref_img_no, int which_blur_sample)
{
    char text_file_name[60];
    sprintf(text_file_name,"../data/scene_%02d_%04d.txt",which_blur_sample,(int)ref_img_no);

    cout << "text_file_name = " << text_file_name << endl;

    ifstream cam_pars_file(text_file_name);

    char readlinedata[300];

    float4 direction;
    float4 upvector;
    Vector<3>posvector;


    while(1)
    {
        cam_pars_file.getline(readlinedata,300);

        if ( cam_pars_file.eof())
            break;


        istringstream iss;


        if ( strstr(readlinedata,"cam_dir")!= NULL)
        {


            string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction.x ;
            iss.ignore(1,',');
            iss >> direction.y ;
            iss.ignore(1,',') ;
            iss >> direction.z;
            iss.ignore(1,',');
            cout << direction.x<< ", "<< direction.y << ", "<< direction.z << endl;
            direction.w = 0.0f;

        }

        if ( strstr(readlinedata,"cam_up")!= NULL)
        {

            string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


            iss.str(cam_up_str);
            iss >> upvector.x ;
            iss.ignore(1,',');
            iss >> upvector.y ;
            iss.ignore(1,',');
            iss >> upvector.z ;
            iss.ignore(1,',');


            upvector.w = 0.0f;

        }

        if ( strstr(readlinedata,"cam_pos")!= NULL)
        {
//            cout<< "cam_pos is present!"<<endl;

            string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

//            cout << "cam pose str = " << endl;
//            cout << cam_pos_str << endl;

            iss.str(cam_pos_str);
            iss >> posvector[0] ;
            iss.ignore(1,',');
            iss >> posvector[1] ;
            iss.ignore(1,',');
            iss >> posvector[2] ;
            iss.ignore(1,',');
//            cout << posvector[0]<< ", "<< posvector[1] << ", "<< posvector[2] << endl;

        }

    }

    /// z = dir / norm(dir)
    Vector<3> z;
    z[0] = direction.x;
    z[1] = direction.y;
    z[2] = direction.z;
    normalize(z);
//    cout << " z = " << z << endl;

    /// x = cross(cam_up, z)
    Vector<3> x = Zeros(3);
    x[0] =  upvector.y * z[2] - upvector.z * z[1];
    x[1] =  upvector.z * z[0] - upvector.x * z[2];
    x[2] =  upvector.x * z[1] - upvector.y * z[0];

    normalize(x);
//    cout << " x = " << x << endl;

    /// y = cross(z,x)
    Vector<3> y = Zeros(3);
    y[0] =  z[1] * x[2] - z[2] * x[1];
    y[1] =  z[2] * x[0] - z[0] * x[2];
    y[2] =  z[0] * x[1] - z[1] * x[0];

//    cout << " y = " << y << endl;

    Matrix<3,3> R = Zeros(3,3);
    R[0][0] = x[0];
    R[1][0] = x[1];
    R[2][0] = x[2];

    R[0][1] = y[0];
    R[1][1] = y[1];
    R[2][1] = y[2];

    R[0][2] = z[0];
    R[1][2] = z[1];
    R[2][2] = z[2];

//    cout << "R = " << R << endl;

    return TooN::SE3<>(R, posvector);

}


void  Fill3Dpoints(float* depth_vals, int ref_img_no, const TooN::Matrix<3>& K, const int width, const int height, double &max_depth)
{

    float u0 = K(0,2);
    float v0 = K(1,2);

    float fx = K(0,0);
    float fy = K(1,1);

    assert(fy<0);

    ifstream ifile;

    char depth_file_name[60];

    sprintf(depth_file_name,"../data/scene_00_%04d.depth",ref_img_no);

    max_depth = -999.0;

    cout << "depth_file_name =" << depth_file_name << endl;

    ifile.open(depth_file_name);

    for(int i = 0 ; i < height ; i++)
    {
        for (int j = 0 ; j < width ; j++)
        {
            double val = 0;
            ifile >> val;

            double u = j;
            double v = i;

            float z =  val / sqrt(((u-u0)/fx)*((u-u0)/fx) + ((v-v0)/fy)*((v-v0)/fy) + 1 ) ;


            depth_vals[i*width+j] = z;

            if ( z > max_depth )
            {
                max_depth = z;
            }
        }
    }
    ifile.close();
}


void get_camera_and_RT(float2& fl, float2& pp, TooN::Matrix<3,3>& R_lr_,
                       TooN::Matrix<3,1>&t_lr_, const unsigned int width,
                       const unsigned int height, bool use_povray,
                       float *depth_vals, int ref_no, int live_no)

{

    TooN::SE3<> PoseRef;
    TooN::SE3<> PoseLive;
    double max_depth = 0;


    if ( use_povray )
    {
        /// Remember to check that MATLAB has coordinates beginning from (1,1) instead of (0,0)
        /// so we should see u0MATLAB = u0Cpp+1 and so is the case with v0

        float KMat[3][3] = {481.2047,         0,  319.5000,
               0,  -480.0000,  239.5000,
               0,         0,    1.0000};

//        float KMat[3][3]=  {3072.0,         0,    319.5,
//                     0,   -3072.0,    239.5,
//                     0, 0,1};

        TooN::Matrix<3> K = Identity(3);

        K(0,0) = KMat[0][0];
        K(0,1) = KMat[0][1];
        K(0,2) = KMat[0][2];

        K(1,0) = KMat[1][0];
        K(1,1) = KMat[1][1];
        K(1,2) = KMat[1][2];

        K(2,0) = KMat[2][0];
        K(2,1) = KMat[2][1];
        K(2,2) = KMat[2][2];



        PoseRef  = computeTpov_cam(ref_no,0);
        PoseLive = computeTpov_cam(live_no,0);

//        float *depth_vals = new float[width*height];

        Fill3Dpoints(depth_vals, ref_no, K, width, height, max_depth);

        fl = make_float2((K(0,0)/640.0f)*width,(K(1,1)/480.0f)*height);
        pp = make_float2((K(0,2)/640.0f)*width,(K(1,2)/480.0f)*height);

    }
    else
    {
        //  float R_wrArr[3][3] = { 0.993199, 0.114383, -0.0217434,
        //  0.0840021, -0.833274, -0.546442,
        //  -0.0806218, 0.540899, -0.837215,};

         fl = make_float2(0.709874*width, 0.945744*height);
         pp = make_float2(0.493648*width, 0.514782*height);


          float R_wrArr[3][3] = {0.993701, 0.110304, -0.0197854,
                               0.0815973, -0.833193, -0.546929,
                               -0.0768135, 0.541869, -0.836945};

          TooN::Matrix<3,3> R_wr = Zeros(3);
          R_wr(0,0) = R_wrArr[0][0];
          R_wr(0,1) = R_wrArr[0][1];
          R_wr(0,2) = R_wrArr[0][2];

          R_wr(1,0) = R_wrArr[1][0];
          R_wr(1,1) = R_wrArr[1][1];
          R_wr(1,2) = R_wrArr[1][2];

          R_wr(2,0) = R_wrArr[2][0];
          R_wr(2,1) = R_wrArr[2][1];
          R_wr(2,2) = R_wrArr[2][2];


          TooN::Vector<3> t_wr;
          t_wr[0] = 0.280643; /*0.295475;*/
          t_wr[1] = -0.255355;/*-0.25538;*/
          t_wr[2] = 0.810979; /*0.805906;*/

          PoseRef = TooN::SE3<>(R_wr,t_wr);


          float R_wlArr[3][3] = {0.993479, 0.112002, -0.0213286,
                                 0.0822353, -0.83349, -0.54638,
                                 -0.0789729, 0.541063, -0.837266};

          TooN::Matrix<3,3> R_wl = Zeros(3);
          R_wl(0,0) = R_wlArr[0][0];
          R_wl(0,1) = R_wlArr[0][1];
          R_wl(0,2) = R_wlArr[0][2];

          R_wl(1,0) = R_wlArr[1][0];
          R_wl(1,1) = R_wlArr[1][1];
          R_wl(1,2) = R_wlArr[1][2];

          R_wl(2,0) = R_wlArr[2][0];
          R_wl(2,1) = R_wlArr[2][1];
          R_wl(2,2) = R_wlArr[2][2];


          TooN::Vector<3> t_wl;
          t_wl[0] = 0.287891;
          t_wl[1] = -0.255839;
          t_wl[2] = 0.808608;


          PoseLive = TooN::SE3<>(R_wl,t_wl);

    }

    cout << "Refpose translation  = " << PoseRef.get_translation() << endl;
    cout << "Livepose translation = " << PoseLive.get_translation() << endl;

    cout << "Refpose Rotation = " << PoseRef.get_rotation() << endl;
    cout << "Livepose Rotation = " << PoseLive.get_rotation() << endl;

    TooN::SE3<> T_lr = PoseLive.inverse()*PoseRef;
    R_lr_ = T_lr.get_rotation().get_matrix();
    t_lr_ = T_lr.get_translation().as_col();

    cout << "max_depth = " << max_depth << endl;
    if ( use_povray )
    {
        t_lr_ = t_lr_  / max_depth;

        for(int i = 0 ; i < width*height ; i++)
        {
            depth_vals[i] = depth_vals[i]/max_depth;
        }
    }

}

int main( int /*argc*/, char* argv[] )
{

  cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
  pangolin::CreateGlutWindowAndBind("Main",512*2+150 ,512*2);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glewInit();


  View& d_panel = pangolin::CreatePanel("ui")
          .SetBounds(1.0, 0.0, 0, Attach::Pix(150));

  bool use_povray = false;
  bool compute_disparity = true;

//  bool use_diffusion_tensor = true;

//  if ( use_povray )
//      use_diffusion_tensor = false;

  TVL1DepthEstimation *Stereo2D;


//  int ref_no = 34;//453;
//  int live_no = 35;//454;

  int ref_no = 453;
  int live_no = 454;

  char refimgfileName[30];
  char curimgfileName[30];

  sprintf(refimgfileName,"../data/scene_00_%04d.png",ref_no);
  sprintf(curimgfileName,"../data/scene_00_%04d.png",live_no);

  if ( use_povray)
      Stereo2D = new TVL1DepthEstimation(refimgfileName,curimgfileName);
  else
  {
//      Stereo2D = new TVL1DepthEstimation("../data/im4Ankur0_by2.png","../data/im4Ankur1_by2.png");
      Stereo2D = new TVL1DepthEstimation("../data/Baby2/view1_by2.png","../data/Baby2/view0_by2.png");
//      Stereo2D = new TVL1DepthEstimation("../data/lena_sigma25.png" ,"../data/lena_sigma25.png");

  }

  unsigned int width  = Stereo2D->getImgWidth();
  unsigned int height = Stereo2D->getImgHeight();



  float2 fl, pp;

  TooN::Matrix<3,3> R_lr_ = Zeros(3);
  TooN::Matrix<3,1> t_lr_ = Zeros(3);

  iu::ImageCpu_32f_C1 *depth_vals = new iu::ImageCpu_32f_C1(IuSize(width,height));

  iu::ImageGpu_32f_C1 *d_depthvals = new iu::ImageGpu_32f_C1(IuSize(width,height));;

  get_camera_and_RT(fl,pp,R_lr_,t_lr_,width, height, use_povray, depth_vals->data(), ref_no, live_no);

  iu::copy(depth_vals,d_depthvals);


  cout << "Width = "<< width << ", Height = " << height << endl;



  View& view_image0 = Display("image0").SetAspect(width*1.0/height*1.0);
  View& view_image1 = Display("image1").SetAspect(width*1.0/height*1.0);
  View& view_image2 = Display("image2").SetAspect(width*1.0/height*1.0);
  View& view_image3 = Display("image3").SetAspect(width*1.0/height*1.0);


  View& d_imgs = pangolin::Display("images")
                     //.SetBounds(1.0, 0.8, 150, 1.0, false)
                     //first distance from bottom left of opengl window i.e. 0.7 is 70%
                     //co-ordinate from bottom left of screen from
                     //0.0 to 1.0 for top, bottom, left, right.
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

   int when2show = 0;
   int warps = 0;

   cout << "Everything ready to run" << endl;

   // alpha = 5.0 and b = 0.5

   bool have_computed_tensor = false;

   iu::ImageGpu_32f_C1 err = iu::ImageGpu_32f_C1(width,height);
   iu::setValue(0,&err,err.roi());

  while(!pangolin::ShouldQuit())
  {

    static Var<bool> resetsq("ui.Reset Seq",false);

    static Var<bool> use_diffusion_tensor("ui.use_diffusion_tensor",true);

    static Var<float> lambda_("ui.lambda", 0.011, 0, 5);

    float lambda = lambda_;

    static Var<float> sigma_p("ui.sigma_p", 0.5 , 0, 4);
    static Var<float> sigma_q("ui.sigma_q", 0.02 , 0, 4);
    static Var<float> tau("ui.tau", 0.05 , 0, 4);
    static Var<float> HuberEpsilon("ui.HuberEpsilon", 1E-2 , 5E-3, 5E-1);


    static Var<int> max_iterations("ui.max_iterations", 3 , 1, 30);
//    static Var<int> max_warps("ui.max_warps", 20 , 0, 400);


//    static Var<float> u0initval("ui.u0initval", 0.5 , 0, 1);

//#ifdef compute_disparity
    static Var<int> u0initval("ui.u0initval", 4 , -10, 10);
//#endif

    static Var<float> alpha("ui.alpha", 5.0,0,100);
    static Var<float> beta("ui.beta", 0.5, 0,2);

    static Var<float> dmin("ui.dmin", 0.01 , 0, 2);
    static Var<float> dmax("ui.dmax", 1 , 0, 4);


    if(HasResized())
      DisplayBase().ActivateScissorAndClear();



    static long unsigned int iterations = 0;
    iterations++;

    if( (resetsq) )
    {
        warps = 0 ;
        iterations = 0;
        cout << "Going to initialise" << endl;
        Stereo2D->InitialiseVariables(u0initval);
//        iu::copy(Stereo2D->d_ref_image,Stereo2D->d_u);
        Stereo2D->computeDiffusionTensor(alpha,beta);
    }

    {


//        //copy current primal variable into the
//        //linearisation primal variable
        Stereo2D->doOneWarp();
//        //compute the gradient around the current linearisation point
        Stereo2D->computeImageGradient_wrt_depth(fl,
                                                pp,
                                                R_lr_,
                                                t_lr_,
                                                compute_disparity,
                                                dmin,
                                                dmax);

        Stereo2D->computeDiffusionTensor(alpha,beta);


    }


    for(int it = 0 ; it < max_iterations ;it++){

        //update the dual variable of the gradient of the primal: p
         Stereo2D->updatedualReg(lambda,
                               tau,
                               sigma_q,
                               sigma_p,
                               HuberEpsilon,
                               use_diffusion_tensor);


         //update the primal variable u
         //using the newly linearised data term.
         //All iterations it therefore use the same linearisation
         //point u0.
         Stereo2D->updatePrimalData(lambda,
                                tau,
                                sigma_q,
                                sigma_p,
                                use_diffusion_tensor);
    }

    cout << "Max iterations: Done..." << endl;
    cout << "Warp No: " <<warps <<endl;
    //here we warp the second image
    //using the current primal variable u
    //for our consumption.
    Stereo2D->updateWarpedImage(fl,
                               pp,
                               R_lr_,
                               t_lr_,
                               compute_disparity);





    //compute error between the warped image give the solution, and the reference frame.

     iu::addWeighted(Stereo2D->d_ref_image, 1,Stereo2D->d_cur2ref_warped,-1,&err,err.roi() );
     if ( iterations % 1 == 0)
    {
         glColor4f(1.0,1.0,1.0,1.0);
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,Stereo2D->d_ref_image->data(), Stereo2D->d_ref_image->pitch(),pbo,tex_show);

        view_image1.Activate();
       // DisplayFloatDeviceMem(&view_image1,u_disp.data(),u_disp.pitch(),pbo,tex_show);
         //DisplayFloatDeviceMem(&view_image1,Stereo2D->d_u->data(),Stereo2D->d_u->pitch(),pbo,tex_show);

        static Var<float> minVal("ui.minVal", 0 , 0, 10);
        static Var<float> maxVal("ui.maxVal", 8 , 0, 10);

        DisplayFloatDeviceMemNorm(&view_image1,Stereo2D->d_u->data(),Stereo2D->d_u->pitch(),pbo,tex_show,false,true,minVal,maxVal);

//        view_image2.ActivateAndScissor();
        //DisplayFloatDeviceMem(&view_image2,Stereo2D->d_cur2ref_warped->data(),Stereo2D->d_cur2ref_warped->pitch(),pbo,tex_show);
        DisplayFloatDeviceMemNorm(&view_image2,(&err)->data(),(&err)->pitch(),pbo,tex_show,false,true,-1,1);


        view_image3.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image3,d_depthvals->data(),d_depthvals->pitch(),pbo,tex_show);
    }



    GPUMem memory;
    memory.getGPUMem();
//    cout << "Total Memory = " << memory.total_mbyte << "MBs, Free = " << memory.free_mbyte << "MBs, Used = " << memory.used_mbyte << "MBs" << endl;
//    cout << "*** Total Memory  ::: "<< memory.total_mbyte << "MBs" << endl;
//    cout << "***  Free Memory  ::: "<< memory.free_mbyte << "MBs" << endl;
//    cout << "***  Used Memory  ::: "<< memory.used_mbyte << "MBs" << endl;

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();

  }


  return 0;
}


