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

  View* d_img[] = {
    &pangolin::Display("i0").SetAspect(640/480.0),
    &pangolin::Display("i1").SetAspect(640/480.0),
    &pangolin::Display("i3").SetAspect(640/480.0),
    &pangolin::Display("i4").SetAspect(640/480.0)
  };

  View& d_imgs = pangolin::Display("images")
    .SetBounds(1.0, 0, 150, 1.0, false)
    .SetLayout(LayoutEqual)
    .AddDisplay(*d_img[0])
    .AddDisplay(*d_img[1])
    .AddDisplay(*d_img[2])
    .AddDisplay(*d_img[3]);

//  View& d_3d = pangolin::Display("3d")
//    .SetBounds(1.0, 0.0, 150, 1.0, -640/480.0)
//    .SetHandler(new Handler3D(s_cam));

  const ImageRef vidsize = video->size();
  const unsigned width = vidsize.x;
  const unsigned height = vidsize.y;


  GlTextureCudaArray tex_vid(vidsize.x, vidsize.y, GL_RGBA32F);
  GlBufferCudaPtr pbo_debug(GlPixelUnpackBuffer,  width*height*sizeof(float4), cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);
  GlTexture tex_show(width, height, GL_RGBA32F);

  for(int frame_num=0; !pangolin::ShouldQuit();)
  {
    const bool run = continuous || Pushed(step);
    static Var<bool> resetsq("ui.Reset Seq",false,false);
    static Var<bool> step("ui.Step", false, false);
    static Var<bool> continuous("ui.Run", false);
    static Var<uint> esm_max_its("ui.ESM max iterations", 15, 0, 50);
    static Var<double> tracking_err("ui.TErr");


    if(HasResized())
      DisplayBase().ActivateScissorAndClear();

    // Show Images
    glDisable(GL_DEPTH_TEST);
    glColor3f(1.0,1.0,1.0);

    d_img[0]->ActivateAndScissor();
    CopyPboToTex(ref->pbo_im,tex_show, GL_RGBA, GL_FLOAT );
    tex_show.RenderToViewportFlipY();

    d_img[1]->ActivateAndScissor();
    CopyPboToTex(live->pbo_im,tex_show, GL_RGBA, GL_FLOAT );
    tex_show.RenderToViewportFlipY();

    d_img[2]->ActivateAndScissor();
    CopyPboToTex(pbo_debug,tex_show, GL_RGBA, GL_FLOAT );
    tex_show.RenderToViewportFlipY();

    d_panel.Render();
    glutSwapBuffers();
    glutMainLoopEvent();
//    boost::this_thread::sleep(boost::posix_time::milliseconds(10));
  }

  delete live;
  delete ref;

  return 0;
}
