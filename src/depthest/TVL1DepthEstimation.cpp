#include "TVL1DepthEstimation.h"
#include <iostream>
#include "../kernels/primal_dual_update.h"
#include "../kernels/pd_depth.cu"

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

TVL1DepthEstimation::TVL1DepthEstimation(const std::string& refimgfile, const int nimages)
{
    _nimages = nimages;
    ref_file_name = refimgfile;
    if(!allocated)
        allocateMemory(width,height);
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

    /// data term saved e.g. I(d) + (d-d0)*gradI_d0 - Ir
    d_data_term     =  new iu::ImageGpu_32f_C1(width,height);

    /// gradient term saved e.g. gradI_d0
    d_gradient_term =  new iu::ImageGpu_32f_C1(width,height);

    /// Store the warped image
    d_cur2ref_warped = new iu::ImageGpu_32f_C1(width,height);

#ifdef RICHARD_IMPLEMENTATION
    for(int i = 0 ; i < _nimages - 1; i++)
    {
        d_data_derivs.push_back( new iu::ImageGpu_32f_C4 (width,height); );
        d_dual_data.push_back(new iu::ImageGpu_32f_C1(width,height));
        d_data_images.push_back(new iu::ImageGpu_32f_C1(width,height));
    }
    datasum = new iu::ImageGpu_32f_C1(width,height);
#endif

    allocated = true;

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

    }

#ifdef RICHARD_IMPLEMENTATION

    assert(_nimages>1);
    for (int i = 1 ; i <= _nimages-1 ; i++ )
    {
        std::string curfilename(ref_file_name.begin(), ref_file_name.end()-8);
        char fileno[5];
        int ref_file_num = atoi(ref_file_name.substr(ref_file_name.length()-8,4).c_str());
        sprintf(fileno,"%04d",ref_file_num+i);

        curfilename = curfilename + std::string(fileno) + ".png";

        d_data_images.at(i-1) = iu::imread_cu32f_C1(curfilename);

    }

#else
    /// Bind the texture
    BindDepthTexture(d_cur_image->data(),
                     d_cur_image->width(),
                     d_cur_image->height(),
                     d_cur_image->stride());
#endif

}



void TVL1DepthEstimation::updatedualReg(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{


#ifdef RICHARD_IMPLEMENTATION
    /// From Richard!
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);
     updateDualReg<<<grid,blocks>>>(d_u->data(),
                   d_px->data(),
                   d_py->data(),
                   sigma_dual_reg,
                   lambda,
                   0,
                   int2(d_u->width(),d_u->height()),
                   d_u->stride());
#else
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
#endif

}

void TVL1DepthEstimation::updatedualData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{

#ifdef RICHARD_IMPLEMENTATION

    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    for(int i = 0 ; i < _nimages-1;i++)
    {

         updateDualData<<<grid,blocks>>>(d_u->data(),
                        d_data_derivs.at(i)->data(),
                        d_dual_data.at(i)->data(),
                        sigma_dual_data,
                        d_dual_data.at(i)->stride(),
                        d_data_derivs.at(i)->stride());
    }

    iu::setValue(0,datasum,datasum->roi());
    for(int i = 0 ; i < _nimages-1; i++)
    {
            updateDataSum<<<grid,block>>(datasum->data(),
                                         d_data_derivs.at(i)->data(),
                                         d_dual_data.at(i)->data(),
                                         d_dual_data.at(i)->stride(),
                                         d_data_derivs.at(i)->stride());
    }

    float scaleData =1.0f/(_nimages-1);
    iu::mulC(datasum,scaleData,datasum,datasum->roi());


#else
    doOneIterationUpdateDualData( d_q->data(),
                                  d_q->stride(),
                                  d_q->width(),
                                  d_q->height(),
                                  d_data_term->data(),
                                  d_gradient_term->data(),
                                  d_u->data(),
                                  d_u0->data(),
                                  lambda,
                                  sigma_primal,
                                  sigma_dual_data,
                                  sigma_dual_reg
                                 );

#endif

}
void TVL1DepthEstimation::updatePrimalData(const float lambda, const float sigma_primal, const float sigma_dual_data, const float sigma_dual_reg)
{

#ifdef RICHARD_IMPLEMENTATION
    dim3 block(boost::math::gcd<unsigned>(width,32), boost::math::gcd<unsigned>(height,32), 1);
    dim3 grid( width / block.x, height / block.y);

    updateSummedPrimal2Denoise<<<grid,blocks>>>(d_u->data(),
                               datasum->data(),
                               d_px->data(),
                               d_py->data(),
                               sigma_primal,
                               int2(d_u->width(),d_u->height()),
                               d_u->stride());
#else
    doOneIterationUpdatePrimal(   d_u->data(),
                                  d_u0->data(),
                                  d_u->stride(),
                                  d_u->width(),
                                  d_u->height(),
                                  d_data_term->data(),
                                  d_gradient_term->data(),
                                  d_px->data(),
                                  d_py->data(),
                                  d_q->data(),
                                  lambda,
                                  sigma_primal,
                                  sigma_dual_data,
                                  sigma_dual_reg
                                 );
#endif



}

void TVL1DepthEstimation::computeImageGradient_wrt_depth(const float2 fl,
                                                         const float2 pp,
                                                         TooN::Matrix<3,3> R_lr_,
                                                         TooN::Matrix<3,1> t_lr_,
                                                         bool disparity,
                                                         float dmin,
                                                         float dmax
#ifdef RICHARD_IMPLEMENTATION
                                                         , unsigned int which_image
#endif
                                                         )

{

#ifdef RICHARD_IMPLEMENTATION

    cumat<3,3> R = cumat_from<3,3,float>(R_lr_);
    cumat<3,1> t = cumat_from<3,1,float>(t_lr_);

    BindDepthTexture(d_data_images.at(which_image)->data(),
                     d_data_images.at(which_image)->width(),
                     d_data_images.at(which_image)->height(),
                     d_data_images.at(which_image)->stride());

                     cu_compute_dI_dz<<<grid,block>>>(d_data_derivs.at(which_image)->data(),
                                                      R,
                                                      t,
                                                      pp,
                                                      fl,
                                                      d_data_derivs.at(which_image)->stride(),
                                                      d_ref_image->data(),
                                                      d_u0->data(),
                                                      d_ref_image->stride());


#else


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
                                     dmax);

#endif

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
