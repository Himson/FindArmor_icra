PROJECT(FindArmor)         

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
# OpenCV > 3 for KCF tracker
FIND_PACKAGE( OpenCV 3 REQUIRED )
# OpenMP
FIND_PACKAGE( OpenMP)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
SET(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
SET(OpenMP_LIBS gomp)
SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# Declare the target (an executable)
include_directories( ${PROJECT_SOURCE_DIR}/include)
ADD_EXECUTABLE(FindArmor  
    main.cpp
    src/augmented_unscented_kalman.cpp
    src/feature.cpp
    src/featureColorName.cpp
    src/PFSolver.hpp
    src/precomp.hpp
    src/tracker.cpp
    src/trackerFeature.cpp
    src/trackerFeatureSet.cpp
    src/trackerKCF.cpp
    src/trackerModel.cpp
    src/trackerSampler.cpp
    src/trackerSamplerAlgorithm.cpp
    src/trackerStateEstimator.cpp
    src/TrackingFunctionPF.hpp
    src/unscented_kalman.cpp
  )
TARGET_COMPILE_OPTIONS(FindArmor PRIVATE ${OpenMP_FLAGS})
TARGET_LINK_LIBRARIES(FindArmor ${OpenCV_LIBS} ${OpenMP_LIBS} /lib/libMVSDK.so)
#MESSAGE(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")     
