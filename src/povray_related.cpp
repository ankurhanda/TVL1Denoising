#include <GL/glew.h>

#include <boost/thread.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <cvd/videosource.h>
#include <TooN/sl.h>
#include <TooN/so3.h>
#include <TooN/se3.h>
#include <TooN/LU.h>

#define HAVE_TOON

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
