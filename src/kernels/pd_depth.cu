#ifndef _CPP_CUDA_VISION_KERNEL_PRIMALDUAL_DEPTH_CU_
#define _CPP_CUDA_VISION_KERNEL_PRIMALDUAL_DEPTH_CU_

#include "../kernels/derivatives.h"
#include "cumath.h"
#include <cutil_inline.h>

texture<float, 2, cudaReadModeElementType> TexImgCur;

const static cudaChannelFormatDesc chandesc_float1 =
cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

__global__ void updateDualReg(
    /*float * d_ux, float * d_uy,*/
    const float * __restrict__ d_u,
    float * __restrict__ d_px, float * __restrict__ d_py,
//    const float * __restrict__ d_edgeWeight,
    const float sigma, float lambda,float huber_eps,
    const int2 imageSize,const  size_t stridef1){

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,stridef1);


//    const float edgeWeight = d_edgeWeight[indexf1];
//    lambda = lambda*edgeWeight;

    const float u0 = d_u[indexf1];

    const float tau =sigma;//1.0f / (4.0f * sigma + huber_eps);


    const float2 p_n = make_float2(d_px[indexf1],d_py[indexf1]);
    const float2 du_n = make_float2((x == imageSize.x - 1) ? 0: d_u[getImageIdxStrided(x+1,y,stridef1)] - u0,
                                    (y == imageSize.y - 1) ? 0 :d_u[getImageIdxStrided(x,y+1,stridef1)] - u0);


    //gradient descent on the dual variable

    //l1 on grad
    float2 update = (p_n + du_n*tau)/(1+huber_eps*tau);
    const float len =  fmaxf(1,length(update)/lambda);
    //
    //l2 on grad
    //const float len = 1+sigma;
    //float len =  fmaxf(1,powf(length(make_float2(px +  sigma*ux, py + sigma*uy)),30.0f));

    d_px[indexf1] =  update.x/len;
    d_py[indexf1] =  update.y/len;

}


__global__ void updateDualData(
    const float * __restrict__ d_u,
    const float4 * __restrict__ d_derivs_data,
    float *  d_q, const float sigma, float lambda,
    const  size_t stridef1, const size_t stridef4
    ){

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,stridef1);
    const int indexf4 = getImageIdxStrided(x,y,stridef4);

    const float q = d_q[indexf1];
    const float u = d_u[indexf1];
    const float4 data = d_derivs_data[indexf4];
    const float dIdz = data.x;
    const float Id = data.y;
    const float z0 = data.z;
    const float Ir = data.w;


    //l1
    float err = (Id-Ir) + (u - z0)*dIdz ;
    float q_update = q +  sigma*err;//q + lambda*sigma_p.*inner;
    //float reprojection_p = fmaxf(1.0,abs(q));
    //q = q/reprojection_p ;
    const float len =  fmaxf(1,fabs(q_update)/(lambda));

    //l2
    //const float len = (1+sigma)/lambda;
    //float len =  fmaxf(1,powf(length(make_float2(px +  sigma*ux, py + sigma*uy)),2.0f));

    d_q[indexf1] =  (q_update)/len;


}



__global__ void updatePrimal2Denoise(
    float *  d_u,
    const float4 * __restrict__ d_derivs_data,
    float * __restrict__ d_px,   float * __restrict__ d_py,
    const float * __restrict__ d_q,const float tau,
    const int2 imageSize,const  size_t stridef1, const size_t stridef4){
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,stridef1);
    const int indexf4 = getImageIdxStrided(x,y,stridef4);

    const float4 data = d_derivs_data[indexf4];
    const float dIdz = data.x;

    const float div_p = dxm(d_px, x,y,imageSize,stridef1) + dym(d_py,x,y,imageSize,stridef1);
    const  float u =  d_u[indexf1]  + tau*div_p - tau*d_q[indexf1]*dIdz;
    //d = d + sigma_d*(div_q - lambda*(p.*grad_wrt_d));;
    d_u[indexf1] = u;
}


__global__ void updateDataSum(
    float *  d_datasum,
    const float4 * __restrict__ d_derivs_data,
    const float * __restrict__ d_q,
    const size_t stridef1, const size_t stridef4){

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,stridef1);
    const int indexf4 = getImageIdxStrided(x,y,stridef4);

    const float4 data = d_derivs_data[indexf4];
    const float dIdz = data.x;

    d_datasum[indexf1] = d_datasum[indexf1]+ d_q[indexf1]*dIdz;
}


__global__ void updateSummedPrimal2Denoise(
    float *  d_u,
    float *  d_datasum,
    float * __restrict__ d_px,   float * __restrict__ d_py,
    const float tau,
    const int2 imageSize,const  size_t stridef1){
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,stridef1);

    const float div_p = dxm(d_px, x,y,imageSize,stridef1) + dym(d_py,x,y,imageSize,stridef1);
    const  float u =  d_u[indexf1]  + tau*div_p - tau*d_datasum[indexf1];

    d_u[indexf1] = u;
}


__global__ void cu_compute_dI_dz(float4 *  derivs,
                                       cumat<3,3>R,
                                       cumat<3,1>t,
                                       const float2 d_pp, const float2 d_fl,
//                                       const float2 depth_range,
                                       const size_t stridef4,
                                     float *d_ref_img,
                                        float *d_u0,
                                     const size_t stridef1
                                      )
{




         const unsigned short x = (blockIdx.x*blockDim.x + threadIdx.x);
         const unsigned short y = (blockIdx.y*blockDim.y + threadIdx.y);

         const float2 invfl = 1.0f/d_fl;
         const float3 uvnorm =  make_float3( (x-d_pp.x)*invfl.x,(y-d_pp.y)*invfl.y,1);//*depth_range.y;
         cumat<3,1> uvnormMat = {uvnorm.x, uvnorm.y, uvnorm.z};


         float zLinearised = d_u0[y*stridef1+x];
         zLinearised = fmaxf(0.0f,fminf(1.0f,zLinearised));

//         float3 p3d = uvnorm*zLinearised;

         cumat<3,1> p3d_r       = {uvnorm.x*zLinearised, uvnorm.y*zLinearised, zLinearised};



         /// Are we really sure of this?
         cumat<3,1> p3d_dest  =  R*p3d_r + t;

         float3 p3d_dest_vec = {p3d_dest(0,0), p3d_dest(1,0), p3d_dest(2,0)};
         float2 p2D_live = {p3d_dest(0,0)/p3d_dest(2,0) , p3d_dest(1,0)/p3d_dest(2,0)};

         p2D_live.x= p2D_live.x*d_fl.x + d_pp.x;
         p2D_live.y= p2D_live.y*d_fl.y + d_pp.y;


         float Ir =  d_ref_img[y*stridef1+x];

         float Id =  tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f);
         float Idx = tex2D(TexImgCur,  p2D_live.x+0.5f+1.0f,p2D_live.y+0.5f);
         float Idy = tex2D(TexImgCur,  p2D_live.x+0.5f,p2D_live.y+0.5f+1.0f);


         //dI/dx spatial derivative in x and y
         float2 dIdx = make_float2(Idx-Id, Idy-Id);


         //dK.pi/dX
//         float3 dpi_u = make_float3(d_fl.x/p3d_dest.z, 0, -(d_fl.x*p3d_dest.x)/(p3d_dest.z*p3d_dest.z));
//         float3 dpi_v = make_float3(0, d_fl.y/p3d_dest.z, -(d_fl.y*p3d_dest.y)/(p3d_dest.z*p3d_dest.z));

         //dX/dz d(R.kinv.[x,y,1]z + t)/dz =  R.kinv.[x,y,1] .
         //using z in linear depth
         //float3 dXdz =  multiplySO3(T_k_ref_,uvnorm);
         //using z in inverse depth
//         float3 dXdz =  multiplySO3(T_k_ref_,uvnorm);//*(-1.0f/(zLinearised*zLinearised));

         //chain rule
//         float dIdz =  dot(dIdx, make_float2( dot(dXdz,dpi_u),  dot(dXdz,dpi_v) ) );
//         derivs[y*stridef4 + x]  = make_float4(dIdz,Id,zLinearised,Ir);


         p3d_dest_vec.z = p3d_dest_vec.z + 10E-6;

 //        float3 dpi_u = make_float3(1/p3d_dest_vec.z, 0,-(p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
 //        float3 dpi_v = make_float3(0, 1/p3d_dest_vec.z,-(p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

         float3 dpi_u = make_float3(d_fl.x/p3d_dest_vec.z, 0,-(d_fl.x*p3d_dest_vec.x)/(p3d_dest_vec.z*p3d_dest_vec.z));
         float3 dpi_v = make_float3(0, d_fl.y/p3d_dest_vec.z,-(d_fl.y*p3d_dest_vec.y)/(p3d_dest_vec.z*p3d_dest_vec.z));

         cumat<3,1> dXdz = R*uvnormMat;///(zLinearised*zLinearised);

         float3 dXdz_vec = {dXdz(0,0),dXdz(1,0),dXdz(2,0)};

         float dIdz =  dot(dIdx, make_float2( dot(dXdz_vec,dpi_u),  dot(dXdz_vec,dpi_v) ) );

         derivs[y*stridef4+x] = make_float4(dIdz,Id,zLinearised,Ir);
}


void BindDepthTexture(float* cur_img,
                      unsigned int width,
                      unsigned int height,
                      unsigned int imgStride)

{
    cudaBindTexture2D(0,TexImgCur,cur_img,chandesc_float1,width, height,imgStride*sizeof(float));

    TexImgCur.addressMode[0] = cudaAddressModeClamp;
    TexImgCur.addressMode[1] = cudaAddressModeClamp;
    TexImgCur.filterMode = cudaFilterModeLinear;
    TexImgCur.normalized = false;    // access with normalized texture coordinates
}

//void TVL1Depth::init(){
//    std::cout << "Init "  <<std::endl;
//    iu::setValue(0,dual_reg_x,dual_reg_x->roi());
//    iu::setValue(0,dual_reg_y,dual_reg_x->roi());

//    iu::setValue(0.5,primal_reg,primal_reg->roi());

////    iu::ImageCpu_32f_C1 randim(primal_reg->width(),primal_reg->height());
////    float * d = randim.data();
////    for(int i =0 ; i < randim.numel();i++){
////        d[i] = (float)drand48();
////    }
////    iu::copy(&randim,primal_reg);

//    for(int k = 0 ; k < dual_data.size();k++){
//        iu::setValue(0.5,dual_data.at(k),dual_data.at(k)->roi());
//    }
//    //iu::copy(this->data_init,this->primal_reg);
//    //iu::copy(this->data_init,this->dual_data);

//    //       iu::filterGauss(data,this->ureg,data->roi(),1);
//    //        iu::filterGauss(data,this->udata,data->roi(),1);
//}



//void TVL1Depth::computeMultiDepth(std::vector<FrameData *> & dataTerms, int reference, bool derivCheck){
//    //ScopedCuTimer time("denoise");

//    int2 imageSize_ = make_int2(primal_reg->width(),primal_reg->height());
//    dim3 blockdim(boost::math::gcd<unsigned>(imageSize_.x,32), boost::math::gcd<unsigned>(imageSize_.y,32), 1);
//    dim3 griddim( imageSize_.x / blockdim.x, imageSize_.y / blockdim.y);

//    for(int o = 0 ; o<outerIterations;o++){

//        //compute using the current linearisation point
//        //derivatives and image prediction.
//        FrameData * frameRef =  dataTerms[reference];
//        for(int k = 0 ; k < dataTerms.size();k++ ){
//            if(k!=reference){
//                FrameData * frameData =  dataTerms[k];
//                mvs->compute_dI_dz(mvs->camera,
//                                   frameRef->grey_st_32f,frameRef->h_T_wf,
//                                   frameData->grey_st_32f,frameData->h_T_wf,
//                                   primal_reg,data_derivs.at(k), derivCheck
//                                   );
//            }
//        }

//        for(int i = 0 ; i < innerIterations; i++)
//        {


//            //update dual variable reg
//            updateDualReg<<<griddim,blockdim>>>(primal_reg->data(),dual_reg_x->data(),dual_reg_y->data(),nrg->data(),sigma_dual,1,epsilon,imageSize_,
//                                                primal_reg->stride());

//            //update each dual variable data
//            for(int k = 0 ; k < dataTerms.size();k++ ){
//                if(k!=reference){
//                    updateDualData<<<griddim,blockdim>>>(primal_reg->data(),data_derivs.at(k)->data(), dual_data.at(k)->data(), sigma_dual, 1/lambda, primal_reg->stride(),
//                                                         dual_data.at(k)->stride());
//                }
//            }

//            //update primal variable
//            //            updatePrimal2Denoise<<<griddim,blockdim>>>(primal_reg->data(),data_derivs->data(),dual_reg_x->data(),dual_reg_y->data(),dual_data->data(),
//            //                                                       sigma_data,imageSize_,primal_reg->stride(),data_derivs->stride());
//            iu::setValue(0,datasum,datasum->roi());
//            for(int k = 0 ; k < dataTerms.size();k++ ){
//                if(k!=reference){
//                    updateDataSum<<<griddim,blockdim>>>(datasum->data(), data_derivs.at(k)->data(), dual_data.at(k)->data(),
//                                primal_reg->stride(), dual_data.at(k)->stride());

//                }
//            }
//            float scaleData =1.0f/dataTerms.size();
//            iu::mulC(datasum,scaleData,datasum,datasum->roi());


//            updateSummedPrimal2Denoise<<<griddim,blockdim>>>(primal_reg->data(),datasum->data(),dual_reg_x->data(),dual_reg_y->data(), sigma_data,imageSize_,primal_reg->stride() );


//        }
//    }
//}

//void TVL1Depth::computeDepth(std::vector<FrameData *> & dataTerms, int reference, int dataterm){

//}

///*void TVL1Depth::computeDepth(std::vector<FrameData *> & dataTerms, int reference, int dataterm){
//    //ScopedCuTimer time("denoise");

//    int2 imageSize_ = make_int2(primal_reg->width(),primal_reg->height());
//    dim3 blockdim(boost::math::gcd<unsigned>(imageSize_.x,32), boost::math::gcd<unsigned>(imageSize_.y,32), 1);
//    dim3 griddim( imageSize_.x / blockdim.x, imageSize_.y / blockdim.y);

//    for(int o = 0 ; o<outerIterations;o++){

//        //compute using the current linearisation point
//        //derivatives and image prediction.
//        FrameData * frameRef =  dataTerms[reference];
//        FrameData * frameData =  dataTerms[dataterm];
//        mvs->compute_dI_dz(mvs->camera,
//                           frameRef->grey_st_32f,frameRef->h_T_wf,
//                           frameData->grey_st_32f,frameData->h_T_wf,
//                           primal_reg,data_derivs
//                           );

//        for(int i = 0 ; i < innerIterations; i++)
//        {


//            //update dual variable reg
//            updateDualReg<<<griddim,blockdim>>>(primal_reg->data(),dual_reg_x->data(),dual_reg_y->data(),nrg->data(),sigma_dual,1,epsilon,imageSize_,
//                                                primal_reg->stride());

//            //update dual variable data
//            updateDualData<<<griddim,blockdim>>>(primal_reg->data(),data_derivs->data(), dual_data->data(), sigma_dual, 1/lambda, primal_reg->stride(),
//                                                 dual_data->stride());

//            //update primal variable
//            updatePrimal2Denoise<<<griddim,blockdim>>>(primal_reg->data(),data_derivs->data(),dual_reg_x->data(),dual_reg_y->data(),dual_data->data(),
//                                                       sigma_data,imageSize_,primal_reg->stride(),data_derivs->stride());


//        }
//    }
//}*/

//void TVL1Depth::updateEdge(iu::ImageGpu_32f_C1 * image){
//    iu::filterEdge(image,
//                   this->nrg,
//                   this->nrg->roi(),
//                   this->alpha,this->beta,0.0001);
//}

//void TVL1Depth::updateEdge(iu::ImageGpu_32f_C4 * image){
//    iu::filterEdge(image,
//                   this->nrg,
//                   this->nrg->roi(),
//                   this->alpha,this->beta,0.0001);
//}

//void TVL1Depth::updateImage(iu::ImageGpu_8u_C1 *image){
//    iu::convert_8u32f_C1(image,image->roi(),this->data_init,this->data_init->roi());
//    //iu::copy(image,this->data);
//}


//void TVL1Depth::updateImage(iu::ImageGpu_32f_C1 *image){
//    // iu::convert_8u32f_C1(image,image->roi(),this->data,this->data->roi());

//    iu::copy(image,this->data_init);

//}


#endif
