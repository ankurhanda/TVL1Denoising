##############################################################################
# CUDA configuration
PROJECT("TVL1Denoising")


CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
ENDIF()

# Platform configuration vars
INCLUDE(SetPlatformVars)



#SET(BUILD_SHARED_LIBS TRUE)
#SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
#SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)


#######################################################
## Find required dependencies
FIND_PACKAGE(FREEGLUT REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(Threads REQUIRED)
#FIND_PACKAGE(freenect REQUIRED)
#FIND_PACKAGE(freenectsync REQUIRED)
FIND_PACKAGE(Boost 1.40 COMPONENTS thread filesystem REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(CVD REQUIRED)
#FIND_PACKAGE(Collada)
#FIND_PACKAGE(Bullet)

#######################################################
## Sources

SET(
  SOURCES
  ./src/kernels/tvl1_denoising.cu
  ./src/utils.h
)




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
  libcutil_x86_64.a
    -lGVars3

)

LINK_DIRECTORIES(
  ${CUDA_SDK_ROOT_DIR}/lib
  ${CUDA_SDK_ROOT_DIR}/common/lib
  ${Boost_LIBRARY_DIRS}
)

# We need shared memory atomics "--maxrregcount=32""--ptxas-options=--opt-level=100"
SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"  "--ptxas-options=--verbose" "-keep"  )
ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION)

#SET(CMAKE_CXX_FLAGS "-Wall")


# compile an .so for the cuda kernel files
#CUDA_ADD_LIBRARY(cudaVision SHARED
#    ${SOURCES}
#)

#
#Cuda Normals
#
#CUDA_ADD_EXECUTABLE(NormalsCuda
#   ${SOURCES}
#  ./src/main/NormalsCuda/main.cpp
#)

#
# DTAM
#

CUDA_ADD_EXECUTABLE(TVL1Denoising
   ${SOURCES}
  ./src/main.cpp
)
