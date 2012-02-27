##############################################################################
# CUDA configuration
PROJECT("TVL1Denoising")


CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

# Platform configuration vars
INCLUDE(SetPlatformVars)

# Make sure your source doesn't get compiled until you fix the warnings!
#ADD_DEFINITIONS( "-Wall -ansi -pedantic" )
ADD_DEFINITIONS( "-O3" )


#SET(BUILD_SHARED_LIBS TRUE)
#SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
#SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)


#######################################################
## Find required dependencies
FIND_PACKAGE(FREEGLUT REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(Boost 1.40 COMPONENTS thread filesystem REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(CVD REQUIRED)
FIND_PACKAGE(MinimalImgUtilities REQUIRED)

#FIND_PACKAGE(Collada)
#FIND_PACKAGE(Bullet)

#######################################################
## Sources




SET(
  SOURCES
  ./src/kernels/tvl1_denoising.cu
  ./src/kernels/disparity1D.cu
  ./src/kernels/Strumdepthestimation.cu
  ./src/kernels/Strumdepthestimation.h
  ./src/kernels/primal_dual_update.h
  ./src/kernels/primal_dual_update.cu
  ./src/kernels/cumath.h
  ./src/kernels/pd_depth.cu
  ./src/kernels/derivatives.h
  ./src/utils.h
)


LIST(APPEND CLASSES TVL1DepthEstimation)

FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ./src/depthest/${class}.cpp ./src/depthest/${class}.h)
ENDFOREACH(class)


#set( CMAKE_VERBOSE_MAKEFILE on )


CUDA_INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
#	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
  )

# C++ only includes
INCLUDE_DIRECTORIES(
        #${CUDA_SDK_ROOT_DIR}/C/common/inc
	${CUDA_CUT_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_BINARY_DIR} # for configuration files
	)


INCLUDE_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/common/inc
  ${Pangolin_INCLUDE_DIR}
  ${CVD_INCLUDE_DIR}
  ${MinimalImgUtilities_INCLUDE_DIR}
)


LINK_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/lib
  ${CUDA_SDK_ROOT_DIR}/common/lib
  ${Boost_LIBRARY_DIRS}

)


LINK_LIBRARIES(
  ${LIBRARY_NAME}
  ${GLEW_LIBRARY}
  ${CG_LIBRARY}
  ${CG_GL_LIBRARY}
  ${Boost_LIBRARIES}
  ${Pangolin_LIBRARY}
  ${CVD_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
  ${FREEGLUT_LIBRARY}
  ${MinimalImgUtilities_LIBRARY}
  libcutil_x86_64.a
   -lGVars3
)


# We need shared memory atomics "--maxrregcount=32""--ptxas-options=--opt-level=100"
SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"  "--ptxas-options=--verbose" "-keep"  )
ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION)


CUDA_ADD_EXECUTABLE(TVL1Denoising
   ${SOURCES}
  ./src/main.cpp
)
