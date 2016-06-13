
include_directories(${PROJECT_SOURCE_DIR}/libviennacl/include/)

if(ENABLE_CUDA)

  if(ENABLE_OPENCL)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DVIENNACL_WITH_OPENCL") #set flags before setting executable!
    cuda_add_library(viennacl SHARED src/backend.cu
                                     src/blas1.cu src/blas1_host.cu src/blas1_cuda.cu src/blas1_opencl.cu
                                     src/blas2.cu src/blas2_host.cu src/blas2_cuda.cu src/blas2_opencl.cu
                                     src/blas3.cu src/blas3_host.cu src/blas3_cuda.cu src/blas3_opencl.cu)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_CUDA")
    target_link_libraries(viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    cuda_add_library(viennacl SHARED src/backend.cu
                                     src/blas1.cu src/blas1_host.cu src/blas1_cuda.cu
                                     src/blas2.cu src/blas2_host.cu src/blas2_cuda.cu
                                     src/blas3.cu src/blas3_host.cu src/blas3_cuda.cu)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_CUDA")
  endif(ENABLE_OPENCL)
else(ENABLE_CUDA)
  if(ENABLE_OPENCL)
    add_library(viennacl SHARED src/backend.cpp
                                src/blas1.cpp src/blas1_host.cpp src/blas1_opencl.cpp
                                src/blas2.cpp src/blas2_host.cpp src/blas2_opencl.cpp
                                src/blas3.cpp src/blas3_host.cpp src/blas3_opencl.cpp)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    add_library(viennacl SHARED src/backend.cpp
                                src/blas1.cpp src/blas1_host.cpp
                                src/blas2.cpp src/blas2_host.cpp
                                src/blas3.cpp src/blas3_host.cpp)
  endif(ENABLE_OPENCL)
endif(ENABLE_CUDA)

# Special linkage for OpenMP under MinGW:
if(ENABLE_OPENMP AND MINGW)
   target_link_libraries(viennacl gomp)
endif(ENABLE_OPENMP AND MINGW)
