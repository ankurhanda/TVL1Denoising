#include "TVL1DepthEstimation.h"
#include <iostream>
#include "../kernels/primal_dual_update.h"
#include <cvd/image.h>
#include <cvd/image_io.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <string>

using namespace std;

TVL1DepthEstimation::TVL1DepthEstimation(const std::string& refimgfile, const std::string& curimgfile)
                    :allocated(false)
{

    d_ref_image = iu::imread_cu32f_C1(refimgfile);
    d_cur_image = iu::imread_cu32f_C1(curimgfile);

    assert( d_ref_image->width() == d_cur_image->width() );
    std::cout << "width = "<< d_ref_image->width() << std::endl;

    const unsigned int width  = d_ref_image->width();
    const unsigned int height = d_ref_image->height();

    if (!allocated)
        allocateMemory(width,height);

    assert(allocated==true);

    InitialiseVariables(0.5);

}

void TVL1DepthEstimation::allocateMemory(const unsigned int width, const unsigned int height)
{

    /// dual regulariser
    d_px      =  new iu::ImageGpu_32f_C1(width,height);
    d_py      =  new iu::ImageGpu_32f_C1(width,height);

    /// dual data
    d_q      =  new iu::ImageGpu_32f_C1(width,height);

    /// primal variable
    d_u        =  new iu::ImageGpu_32f_C1(width,height);

    /// primal variable initialisation
    d_u0     =  new iu::ImageGpu_32f_C1(width,height);

    d_data_term = new iu::VolumeGpu_32f_C1(width,height,_nimages-1);

    d_gradient_term = new iu::VolumeGpu_32f_C1(width,height,_nimages-1);

    d_sortedindices = new iu::VolumeGpu_8u_C1(width,height,_nimages-1);


    /// Store the warped image
    d_cur2ref_warped = new iu::ImageGpu_32f_C1(width,height);

    d_temp_storage = new iu::ImageGpu_32f_C1(width,height);

    for(int i = 0 ; i < _nimages - 1; i++)
    {
        iu::ImageGpu_32f_C1* d_img = new iu::ImageGpu_32f_C1 (width,height);
        d_dataimages.push_back( d_img );
    }

    for(int i = 0 ; i < _nimages - 1; i++)
    {
        iu::ImageGpu_32f_C1* d_dual_var_data = new iu::ImageGpu_32f_C1 (width,height);
        d_dual_data.push_back( d_dual_var_data );
    }


    d_sum_dual_times_grad    =  new iu::ImageGpu_32f_C1(width,height);


    allocated = true;

    std::cout << "Memory has been allocated!" <<std::endl;

}


void TVL1DepthEstimation::InitialiseVariables(float initial_val=0.5)
{

    if (allocated)
    {
        std::cout << "Entering "<<std::endl;

        iu::setValue(0, d_px  , d_px->roi());
        std::cout <<"stride = " << d_px->stride() << std::endl;

        iu::setValue(0, d_py  , d_py->roi());
        std::cout <<"stride = " << d_px->stride() << std::endl;

        iu::setValue(0.0, d_q , d_q->roi());
        std::cout <<"stride = " << d_q->stride() << std::endl;

        iu::setValue(0.0, d_data_term , d_data_term->roi());
        std::cout <<"stride = " << d_data_term->stride() << std::endl;

        iu::setValue(0.0, d_gradient_term , d_gradient_term->roi());
        std::cout <<"stride = " << d_gradient_term->stride() << std::endl;

        iu::setValue(initial_val, d_u0 , d_u0->roi());
        std::cout <<"stride = " << d_u0->stride() << std::endl;

        iu::setValue(initial_val, d_u    , d_u->roi());
        std::cout <<"stride = " << d_u->stride() << std::endl;

        iu::setValue(0.0, d_sum_dual_times_grad , d_sum_dual_times_grad->roi());
        std::cout <<"stride = " << d_sum_dual_times_grad->stride() << std::endl;

    }

    /// Bind the texture
    BindDepthTexture(d_cur_image->data(),
                     d_cur_image->width(),
                     d_cur_image->height(),
                     d_cur_image->stride());

}


void TVL1DepthEstimation::InitialiseVariablesAndImageStack(float initial_val=0.5)
{

    if (allocated)
    {
        std::cout << "Entering "<<std::endl;

        iu::setValue(0, d_px  , d_px->roi());
        std::cout <<"stride = " << d_px->stride() << std::endl;

        iu::setValue(0, d_py  , d_py->roi());
        std::cout <<"stride = " << d_px->stride() << std::endl;

        iu::setValue(0.0, d_q , d_q->roi());
        std::cout <<"stride = " << d_q->stride() << std::endl;

        iu::setValue(initial_val, d_u0 , d_u0->roi());
        std::cout <<"stride = " << d_u0->stride() << std::endl;

        iu::setValue(initial_val, d_u    , d_u->roi());
        std::cout <<"stride = " << d_u->stride() << std::endl;

        iu::setValue(0.0,d_data_term,d_data_term->roi());
        std::cout <<"stride = " << d_data_term->stride() << std::endl;
        std::cout << "slice stride = "<< d_data_term->slice_stride() << endl;

        iu::setValue(0.0f,d_gradient_term,d_gradient_term->roi());
        std::cout <<"stride = " << d_gradient_term->stride() << std::endl;
        std::cout << "slice stride = "<< d_gradient_term->slice_stride() << endl;

        iu::setValue(0, d_temp_storage  , d_temp_storage->roi());
        std::cout <<"stride = " << d_temp_storage->stride() << std::endl;


    }
    else
    {
        cout<< "Can't initialise because memory isn't allocated yet!" << endl;
        exit(EXIT_FAILURE);
    }

}

TVL1DepthEstimation::TVL1DepthEstimation(const std::string& refimgfile, const int nimages)
    :allocated(false)
{
    _nimages = nimages;
    populateImageStack(refimgfile);
}

void TVL1DepthEstimation::populateImageStack(const std::string& refimgfile)
/// nimages includes refimages

{
    d_ref_image = iu::imread_cu32f_C1(refimgfile);
    std::cout << "width = "<< d_ref_image->width() << std::endl;

    const unsigned int width  = d_ref_image->width();
    const unsigned int height = d_ref_image->height();

    if (!allocated)
        allocateMemory(width,height);

    assert(allocated==true);
    InitialiseVariablesAndImageStack(0.5);
    cout << "Variables have been initialised" << endl;

    cout << "_nimages = " << _nimages << endl;
    for(int i = 1 ; i <= _nimages-1 ; i++)
    {
        std::string curfilename(refimgfile.begin(), refimgfile.end()-8);

        char fileno[5];
        int ref_file_num = atoi(refimgfile.substr(refimgfile.length()-8,4).c_str());
        sprintf(fileno,"%04d",ref_file_num+i);

        curfilename = curfilename + std::string(fileno) + ".png";

        d_dataimages.at(i-1) = iu::imread_cu32f_C1(curfilename);

    }


}


void TVL1DepthEstimation::InitialiseWithThisDepthMap(iu::ImageCpu_32f_C1 depth_vals)
{
        iu::copy(&depth_vals,d_u0);
        iu::copy(&depth_vals,d_u);
}


void TVL1DepthEstimation::updatedualReg(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{
    doOneIterationUpdateDualReg( d_px->data(),
                                 d_py->data(),
                                 d_u->data(),
                                 d_u->stride(),
                                 d_px->width(),
                                 d_px->height(),
                                 lambda,
                                 sigma_primal,
                                 sigma_dual_data,
                                 sigma_dual_reg
                                );
}

void TVL1DepthEstimation::updatedualData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{



    for(int i = 0 ; i < _nimages-1; i++)
    {
        doOneIterationUpdateDualData( d_dual_data.at(i)->data(),
                                      d_dual_data.at(i)->stride(),
                                      d_dual_data.at(i)->width(),
                                      d_dual_data.at(i)->height(),
                                      d_data_term->data(),
                                      d_gradient_term->data(),
                                      d_u->data(),
                                      d_u0->data(),
                                      lambda,
                                      sigma_primal,
                                      sigma_dual_data,
                                      sigma_dual_reg,
                                      i,
                                      d_data_term->slice_stride()
                                      );

    }

    iu::setValue(0,d_sum_dual_times_grad,d_sum_dual_times_grad->roi());

    for(int i = 0 ; i < _nimages-1; i++)
    {
        doSumDualTimesGradient(d_dual_data.at(i)->data(),
                               d_dual_data.at(i)->stride(),
                               d_dual_data.at(i)->width(),
                               d_dual_data.at(i)->height(),
                               d_gradient_term->data(),
                               lambda,
                               sigma_primal,
                               sigma_dual_data,
                               sigma_dual_reg,
                               i,
                               d_data_term->slice_stride(),
                               d_sum_dual_times_grad->data());
    }




}
void TVL1DepthEstimation::updatePrimalData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{
    doOneIterationUpdatePrimal(   d_u->data(),
                                  d_u0->data(),
                                  d_u->stride(),
                                  d_u->width(),
                                  d_u->height(),
                                  d_data_term->data(),
                                  d_gradient_term->data(),
                                  d_px->data(),
                                  d_py->data(),
                                  d_sum_dual_times_grad->data(),
                                  lambda,
                                  sigma_primal,
                                  sigma_dual_data,
                                  sigma_dual_reg,
                                  _nimages,
                                  d_data_term->slice_stride(),
                                  d_sortedindices->data());


}


void TVL1DepthEstimation::ObtainImageFromTexture(const int which_image)
{

    obtainImageSlice(which_image,
                     d_temp_storage->data(),
                     d_temp_storage->stride(),
                     getImgWidth(),
                     getImgHeight());

    char fileName[30];
    sprintf(fileName,"img_%03d.png",which_image);
    cout << fileName << endl;
    iu::imsave(d_temp_storage,fileName);


}


void TVL1DepthEstimation::computeImageGradient_wrt_depth(const float2 fl,
                                                         const float2 pp,
                                                         TooN::Matrix<3,3> R_lr_,
                                                         TooN::Matrix<3,1> t_lr_,
                                                         bool disparity,
                                                         float dmin,
                                                         float dmax,
                                                         int which_image)
{



       BindDepthTexture( d_dataimages.at(which_image)->data(),
                         d_dataimages.at(which_image)->width(),
                         d_dataimages.at(which_image)->height(),
                         d_dataimages.at(which_image)->stride());


       doComputeImageGradient_wrt_depth(fl,
                                     pp,
                                     d_u->data(),
                                     d_u0->data(),
                                     d_data_term->data(),
                                     d_gradient_term->data(),
                                     R_lr_,
                                     t_lr_,
                                     d_data_term->stride(),
                                     d_ref_image->data(),
                                     d_ref_image->width(),
                                     d_ref_image->height(),
                                     disparity,
                                     dmin,
                                     dmax,
                                     which_image,
                                     d_data_term->slice_stride());


}

void TVL1DepthEstimation::sortDataterms()
{

    doDatatermSorting( d_data_term->data(),
                       d_gradient_term->data(),
                       d_data_term->slice_stride(),
                       d_data_term->stride(),
                       d_u0->data(),
                       d_u0->stride(),
                       d_sortedindices->data(),
                       d_sortedindices->slice_stride(),
                       d_sortedindices->stride(),
                       getImgWidth(),
                       getImgHeight(),
                       _nimages
                      );

}

void TVL1DepthEstimation::updateWarpedImage ( const float2 fl,
                                              const float2 pp,
                                              TooN::Matrix<3,3> R_lr_,
                                              TooN::Matrix<3,1> t_lr_,
                                             bool disparity)
{
        doImageWarping(fl,
                       pp,
                       R_lr_,
                       t_lr_,
                       d_cur2ref_warped->data(),
                       d_u->data(),
                       d_u->stride(),
                       d_u->width(),
                       d_u->height(),
                       disparity);

}



