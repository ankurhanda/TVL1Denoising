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

#include "utils.h"
//#include "./kernels/tvl1_denoising.cu"

#include "kernels/Strumdepthestimation.h"

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

extern "C" void launch_disparity_kernel_u(float* dpx, float* dpy,float *du, float *du0,float *dq, float *dI1,
                                          float sigma_u, float lambda, unsigned int width,
                                          unsigned int height, unsigned int stride);


extern "C" void launch_disparity_kernel(float* d_I2, unsigned int width, unsigned int height, unsigned int imgStride);

extern "C" void launch_disparity_kernel_copy_u0_to_u(float *du,float *du0, unsigned int width, unsigned int height, unsigned int stride);
extern "C" void launch_disparity_kernel_I2warped(float *dI2warped,float *du, unsigned int width, unsigned int height, unsigned int stride);
//extern "C" void launch_depth_kernel(float* d_I2, unsigned int width, unsigned int height, unsigned int imgStride,
//                                    const Matrix<3,3>& K,
//                                    const Matrix<3,3>& Kinv, const Matrix<3,3>& R_lr );



//extern "C" texture<float, 2, cudaReadModeElementType> my_tex;


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


void  Fill3Dpoints(float* depth_vals, int ref_img_no, const TooN::Matrix<3>& K, const int width, const int height)
{

    float u0 = K(0,2);
    float v0 = K(1,2);

    float fx = K(0,0);
    float fy = K(1,1);

    assert(fy<0);

    ifstream ifile;

    char depth_file_name[60];

    sprintf(depth_file_name,"../data/scene_00_%04d.depth",ref_img_no);

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
        }
    }
    ifile.close();
}




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
//  img_load(Image1, "../data/Baby2/view1.png");
//  img_load(Image1, "../data/scene_00_0338.png");

  img_load(Image1, "../data/im4Ankur0.png");
//  img_load(Image1, "../data/Bowling2/view0.png");

  CVD::Image<float> Image2;
//  img_load(Image2, "../data/Baby2/view2.png");
//  img_load(Image2, "../data/scene_00_0339.png");
    img_load(Image2,"../data/im4Ankur1.png");
//  img_load(Image2, "../data/Bowling2/view1.png");



  /// Remember to check that MATLAB has coordinates beginning from (1,1) instead of (0,0)
  /// so we should see u0MATLAB = u0Cpp+1 and so is the case with v0
  float KMat[3][3] = {481.2047,         0,  319.5000,
         0,  -480.0000,  239.5000,
         0,         0,    1.0000};




  const unsigned width = Image1.size().x;
  const unsigned height = Image1.size().y;

  TooN::Matrix<3> K = Identity(3);
  TooN::Matrix<3> Kinv = Identity(3);


//  TooN::SE3<>PoseRef  = computeTpov_cam(338,0);
//  TooN::SE3<>PoseLive = computeTpov_cam(339,0);


  //  0.993701 0.110304 -0.0197854 0.280643
  //  0.0815973 -0.833193 -0.546929 -0.255355
  //  -0.0768135 0.541869 -0.836945 0.810979

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


  TooN::Vector<3> T_wr;
  T_wr[0] = 0.280643;
  T_wr[1] = -0.255355;
  T_wr[2] = 0.810979;


  TooN::SE3<>PoseRef = TooN::SE3<>(R_wr,T_wr);

//  0.993479 0.112002 -0.0213286 0.287891
//  0.0822353 -0.83349 -0.54638 -0.255839
//  -0.0789729 0.541063 -0.837266 0.808608

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


  TooN::Vector<3> T_wl;
  T_wl[0] = 0.287891;
  T_wl[1] = -0.255839;
  T_wl[2] = 0.808608;


  TooN::SE3<>PoseLive = TooN::SE3<>(R_wl,T_wl);

  cout << "Refpose translation  = " << PoseRef.get_translation() << endl;
  cout << "Livepose translation = " << PoseLive.get_translation() << endl;

  cout << "Refpose Rotation = " << PoseRef.get_rotation() << endl;
  cout << "Livepose Rotation = " << PoseLive.get_rotation() << endl;

  TooN::SE3<> T_lr = PoseLive.inverse()*PoseRef;

//  K(0,0) = KMat[0][0];
//  K(0,1) = KMat[0][1];
//  K(0,2) = KMat[0][2];

//  K(1,0) = KMat[1][0];
//  K(1,1) = KMat[1][1];
//  K(1,2) = KMat[1][2];

//  K(2,0) = KMat[2][0];
//  K(2,1) = KMat[2][1];
//  K(2,2) = KMat[2][2];

//  K = K/2;
//  K(2,2)=1;

//  focal length 0.709874 0.945744
//  principle point 0.493648 0.514782

  K(0,0) =  0.709874*width;
  K(1,1) = 0.945744*height;

  K(0,2) =  0.493648*width;
  K(1,2) =  0.514782*height;

  Kinv(0,0) = 1.0f/K(0,0);
  Kinv(0,2) = -K(0,2)/K(0,0);

  Kinv(1,1) = 1.0f/K(1,1);
  Kinv(1,2) = -K(1,2)/K(1,1);


  cout << K << endl;
  cout << "Kinv " <<endl;
  cout << Kinv << endl;

  cout << "Rotation "<< endl;
  cout << T_lr.get_rotation().get_matrix() << endl;
  cout << T_lr.get_translation() << endl;

  TooN::Vector<3>Kt = K*T_lr.get_translation();

  TooN::Matrix<3,1>KtMat = Zeros(3,1);

  KtMat(0,0) = Kt[0];
  KtMat(1,0) = Kt[1];
  KtMat(2,0) = Kt[2];

//  getchar();

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
  float *data_term;
  float *grad_wrt_d_at_d0;

  float* h_udisp = new float[width*height];
  float* h_pxdisp = new float[width*height];
  float* u0data = new float [width*height];

  cutilSafeCall(cudaMallocPitch(&(dq ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(px ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(py ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(u ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(g ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMallocPitch(&(dI1 ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(dI2 ), &(imagePitchFloat), width* sizeof (float), height));



  cutilSafeCall(cudaMallocPitch(&(udisp ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(pxdisp ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(u0 ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(dI2warped ), &(imagePitchFloat), width* sizeof (float), height));

  cutilSafeCall(cudaMemcpy2D(g, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy2D(dI2, imagePitchFloat, Image2.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMemcpy2D(dI2warped, imagePitchFloat, Image2.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMallocPitch(&(data_term ), &(imagePitchFloat), width* sizeof (float), height));
  cutilSafeCall(cudaMallocPitch(&(grad_wrt_d_at_d0 ), &(imagePitchFloat), width* sizeof (float), height));


//  cutilSafeCall(cudaMemcpy2D(u, imagePitchFloat, Image1.data(), sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
//  cutilSafeCall(cudaMemset(ux,0,sizeof(float)*width*height));
//  cutilSafeCall(cudaMemset(uy,0,sizeof(float)*width*height));

  cutilSafeCall(cudaMemset2D(px,imagePitchFloat,0,width*sizeof(float),height));
  cutilSafeCall(cudaMemset2D(py,imagePitchFloat,0,width*sizeof(float),height));

  cutilSafeCall(cudaMemset2D(dq,imagePitchFloat,0,width*sizeof(float),height));

  cutilSafeCall(cudaMemset2D(data_term,imagePitchFloat,0,width*sizeof(float),height));
  cutilSafeCall(cudaMemset2D(grad_wrt_d_at_d0,imagePitchFloat,0,width*sizeof(float),height));


   unsigned int stride = (unsigned int)(imagePitchFloat/ sizeof(float));
   cout<< "stride = "<<stride <<endl;



   int when2show = 0;
   int warps = 0;


//   Fill3Dpoints(u0data,338,K,width,height);

//   for(int i = 0 ; i < height; i++)
//   {
//       for(int j = 0 ; j < width ; j++)
//       {
//           cout << u0data[i*width+j] << " ";
//       }
//   }

//   cout << endl;
//   getchar();


//   float Kdata[3*3] = {0.1,0.2, 0.3,  0.22, 0.12, 0.32, 0.22, 0.12, 0.23};

//   const Matrix<3,3> KMat = Matrix<3,3>(Kdata,3,3);


//   launch_depth_kernel(dI2, width,height,stride, KMat, KMat, KMat);
//                                       const Matrix<3,3>& K,
//                                       const Matrix<3,3>& Kinv, const Matrix<3,3>& R_lr )



//   cout << "KRKinv = " << K*T_lr.get_rotation().get_matrix()*Kinv << endl;
//   cout << "Kt = " << KtMat << endl;
//   getchar();






  while(!pangolin::ShouldQuit())
  {

    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<bool> step("ui.Step", false, false);
    static Var<bool> continuous("ui.Run", false);
    static Var<double> lambda("ui.lambda", 1, 0, 100);

    static Var<double> sigma_p("ui.sigma_p", 0.5 , 0, 4);
    static Var<double> sigma_q("ui.sigma_q", 0.02 , 0, 4);
    static Var<double> sigma_u("ui.sigma_u", 0.05 , 0, 4);

    static Var<int> max_iterations("ui.max_iterations", 300 , 1, 4000);
    static Var<int> max_warps("ui.max_warps", 20 , 1, 400);

//    static Var<int> u0initval("ui.u0initval", -8 , -100, 100);


    static Var<float> u0initval("ui.u0initval", 0.5 , 0, 1);
    static Var<float> dmin("ui.dmin", 0.1 , 0, 2);
    static Var<float> dmax("ui.dmax", 1 , 0, 4);





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
        cutilSafeCall(cudaMemcpy2D( u,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

    }

    if( Pushed(resetsq))
    {
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
        cutilSafeCall( cudaMemcpy2D(u,  imagePitchFloat, u0data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));


        cutilSafeCall(cudaMemset2D(px,imagePitchFloat,0,width*sizeof(float),height));
        cutilSafeCall(cudaMemset2D(py,imagePitchFloat,0,width*sizeof(float),height));

        cutilSafeCall(cudaMemset2D(dq,imagePitchFloat,0,width*sizeof(float),height));
        cutilSafeCall(cudaMemset2D(dI2warped,imagePitchFloat,0,width*sizeof(float),height));

        cutilSafeCall(cudaMemset2D(data_term,imagePitchFloat,0,width*sizeof(float),height));
        cutilSafeCall(cudaMemset2D(grad_wrt_d_at_d0,imagePitchFloat,0,width*sizeof(float),height));
    }

    if ( iterations > max_iterations)
    {
        iterations = 0;

        cout << "coming here!"<<endl;

        launch_disparity_kernel_copy_u0_to_u(u,u0,width, height,stride);

        warps++;
    }

//    if ( iterations % 100 == 0)
    if ( warps <= max_warps &&  when2show % 1  == 0 )
    {

//   for(int i = 0 ; i < 100; i ++){
//         launch_kernel_derivative_u(ux,uy,u,stride, width , height);
//         launch_kernel_dual_variable_p(px,py,ux,uy,0.5,stride,width,height);
//         launch_disparity_kernel_p (px,py,u,width,height,stride,sigma_p);
//         launch_kernel_dual_variable_q(dq,u,g,1/lambda,lambda, stride,width,height);
//         launch_kernel_update_u(px,py,u,dq,stride,width,height,1/(lambda+4),lambda);
//        launch_disparity_kernel_u (px,py,u,u0,dq,1/(lambda+4),lambda,width,height,stride);


        launch_depth_kernel(dI2,width,height,stride);


        launch_depth_estimation_kernel_compute_data_term_and_gradient(g,
                                                                      data_term,
                                                                      grad_wrt_d_at_d0,
                                                                      u,u0,
                                                                      height,width,stride,
                                                                      K,Kinv,T_lr.get_rotation().get_matrix(),
                                                                      KtMat,dmin,dmax);

        launch_depth_estimation_kernel_q(dq,u,u0,sigma_q,lambda,
                                         g,
                                         width,height, stride,
                                         data_term);

        launch_depth_estimation_kernel_p(px,py,u,
                                         width,height,
                                         stride,sigma_p,
                                         dmin, dmax);


        launch_depth_estimation_kernel_u(px,py,u,u0,dq,
                                         g,
                                         sigma_u,lambda,
                                         width,
                                         height,stride,
                                         dmin, dmax,
                                         grad_wrt_d_at_d0);

        launch_depth_estimation_kernel_I2warped(dI2warped,u,width,height,stride,
                                                K,
                                                Kinv, T_lr.get_rotation().get_matrix(),KtMat,
                                                dmin,
                                                dmax);


          //          launch_kernel_check_grad_wrt_d0_is_correct(u0,
          //                                             height,
          //                                             width,
          //                                             stride,
          //                                             K,
          //                                             Kinv,
          //                                             T_lr.get_rotation().get_matrix(),
          //                                             KtMat);


          //          launch_kernel_check_KRt_is_correct(u0,
          //                                             height,
          //                                             width,
          //                                             stride,
          //                                             K,
          //                                             Kinv,
          //                                             T_lr.get_rotation().get_matrix(),
          //                                             KtMat);

//        launch_disparity_kernel   (dI2,width,height,stride);
//        launch_disparity_kernel_q (dq,u,u0,sigma_q,lambda,g,width,height,stride); // g = I1;
//        launch_disparity_kernel_p (px,py,u,width,height,stride,sigma_p);
//        launch_disparity_kernel_u (px,py,u,u0,dq,g,sigma_u,lambda,width,height,stride); // g = I1;
//        launch_disparity_kernel_I2warped(dI2warped,u,width,height,stride);


            iterations++;


            cudaMemcpy2D(h_udisp,width*sizeof(float),u,imagePitchFloat,width*sizeof(float),height,cudaMemcpyDeviceToHost);
            cudaMemcpy2D(h_pxdisp,width*sizeof(float),px,imagePitchFloat,width*sizeof(float),height,cudaMemcpyDeviceToHost);

            float max_u =-9999.0f;
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

//          cout << "h_udisp[0]" << h_udisp[0] << endl;

            cout << "max_u =" << max_u << ", min_u= "<< min_u << endl;

//          cout << "max_p =" << max_p << ", min_p= "<< min_p << endl;

            cutilSafeCall(cudaMemcpy2D(udisp ,  imagePitchFloat, h_udisp, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
            cutilSafeCall(cudaMemcpy2D(pxdisp,  imagePitchFloat, h_pxdisp, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));

//            CVD::Image<float>image2show(h_udisp,CVD::ImageRef(width,height));
//            CVD::BasicImage<float>image2show(h_udisp,CVD::ImageRef(width,height));
//            img_save(image2show,"name.png");
//            if ( iterations == 100 )
//                getchar();

    }
    when2show++;

    cout << "iterations = " << iterations << endl;


    // if ( iterations % 100 == 0)
    {
        view_image0.Activate();
        DisplayFloatDeviceMem(&view_image0,g,imagePitchFloat,pbo,tex_show);

        view_image1.Activate();
        DisplayFloatDeviceMem(&view_image1,udisp,imagePitchFloat,pbo,tex_show);
//        DisplayFloatDeviceMem(&view_image1,u,imagePitchFloat,pbo,tex_show);

        view_image2.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image2,dI2warped,imagePitchFloat,pbo,tex_show);
//        DisplayFloatDeviceMem(&view_image2,px,imagePitchFloat,pbo,tex_show);

        view_image3.ActivateAndScissor();
        DisplayFloatDeviceMem(&view_image3,py,imagePitchFloat,pbo,tex_show);
    }

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();

  }


  return 0;
}


