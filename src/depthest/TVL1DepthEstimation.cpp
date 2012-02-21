#include "TVL1DepthEstimation.h"
#include <iostream>
#include "../kernels/primal_dual_update.h"

//TVL1DepthEstimation::TVL1DepthEstimation()
//{
//    allocated = false;
//}

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
    d_dual_reg      =  new iu::ImageGpu_32f_C2(width,height);

    /// dual data
    d_dual_data     =  new iu::ImageGpu_32f_C1(width,height);

    /// primal variable
    d_primal        =  new iu::ImageGpu_32f_C1(width,height);

    /// primal variable initialisation
    d_primal_u0     =  new iu::ImageGpu_32f_C1(width,height);

    /// data term saved e.g. I(d) + (d-d0)*gradI_d0 - Ir
    d_data_term     =  new iu::ImageGpu_32f_C1(width,height);

    /// gradient term saved e.g. gradI_d0
    d_gradient_term =  new iu::ImageGpu_32f_C1(width,height);

    /// Store the warped image
    d_cur2ref_warped = new iu::ImageGpu_32f_C1(width,height);

    allocated = true;

}


void TVL1DepthEstimation::InitialiseVariables(float initial_val=0.5)
{

    if (allocated)
    {
        iu::setValue(make_float2(0.0f,0.0f), d_dual_reg  , d_dual_reg->roi());
        iu::setValue(0.0, d_dual_data , d_dual_data->roi());
        iu::setValue(0.0, d_primal    , d_primal->roi());
        iu::setValue(0.0, d_data_term , d_data_term->roi());
        iu::setValue(0.0, d_gradient_term , d_gradient_term->roi());
        iu::setValue(initial_val, d_primal_u0 , d_primal_u0->roi());
    }

    /// Bind the texture
    BindDepthTexture(d_cur_image->data(),
                     d_cur_image->width(),
                     d_cur_image->height(),
                     d_cur_image->stride());

}

void TVL1DepthEstimation::updatedualReg(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{
    doOneIterationUpdateDualReg( d_dual_reg->data(),
                                 d_primal->data(),
                                 d_dual_reg->stride(),
                                 d_dual_reg->width(),
                                 d_dual_reg->height(),
                                 lambda,
                                 sigma_primal,
                                 sigma_dual_data,
                                 sigma_dual_reg
                                );
}

void TVL1DepthEstimation::updatedualData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{

    doOneIterationUpdateDualData( d_dual_data->data(),
                                  d_dual_data->stride(),
                                  d_dual_data->width(),
                                  d_dual_data->height(),
                                  d_data_term->data(),
                                  lambda,
                                  sigma_primal,
                                  sigma_dual_data,
                                  sigma_dual_reg
                                 );

}
void TVL1DepthEstimation::updatePrimalData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{
    doOneIterationUpdatePrimal(   d_primal->data(),
                                  d_primal->stride(),
                                  d_primal->width(),
                                  d_primal->height(),
                                  d_gradient_term->data(),
                                  d_dual_reg->data(),
                                  d_dual_data->data(),
                                  lambda,
                                  sigma_primal,
                                  sigma_dual_data,
                                  sigma_dual_reg
                                 );

}

void TVL1DepthEstimation::computeImageGradient_wrt_depth(const float2 fl,
                                                         const float2 pp,
                                                         TooN::Matrix<3,3> R_lr_,
                                                         TooN::Matrix<3,1> t_lr_)
{




    doComputeImageGradient_wrt_depth(fl,
                                     pp,
                                     d_primal->data(),
                                     d_primal_u0->data(),
                                     d_data_term->data(),
                                     d_gradient_term->data(),
                                     R_lr_,
                                     t_lr_,
                                     d_data_term->stride(),
                                     d_ref_image->data(),
                                     d_ref_image->width(),
                                     d_ref_image->height());

}

void TVL1DepthEstimation::updateWarpedImage ( const float2 fl,
                                              const float2 pp,
                                              TooN::Matrix<3,3> R_lr_,
                                              TooN::Matrix<3,1> t_lr_
                                             )
{
        doImageWarping(fl,
                       pp,
                       R_lr_,
                       t_lr_,
                       d_cur2ref_warped->data(),
                       d_primal->data(),
                       d_primal->stride(),
                       d_primal->width(),
                       d_primal->height());

}
