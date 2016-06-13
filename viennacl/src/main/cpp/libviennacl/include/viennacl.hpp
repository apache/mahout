#ifndef VIENNACL_VIENNACL_HPP
#define VIENNACL_VIENNACL_HPP

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#include <stdlib.h>

#ifdef VIENNACL_WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif


// Extra export declarations when building with Visual Studio:
#if defined(_MSC_VER)
  #if defined(viennacl_EXPORTS)
    #define  VIENNACL_EXPORTED_FUNCTION __declspec(dllexport)
  #else
    #define  VIENNACL_EXPORTED_FUNCTION __declspec(dllimport)
  #endif /* viennacl_EXPORTS */
#else /* defined (_MSC_VER) */
 #define VIENNACL_EXPORTED_FUNCTION
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef int ViennaCLInt;


/************** Enums ***************/

typedef enum
{
  ViennaCLInvalidBackend, // for catching uninitialized and invalid values
  ViennaCLCUDA,
  ViennaCLOpenCL,
  ViennaCLHost
} ViennaCLBackendTypes;

typedef enum
{
  ViennaCLInvalidOrder,  // for catching uninitialized and invalid values
  ViennaCLRowMajor,
  ViennaCLColumnMajor
} ViennaCLOrder;

typedef enum
{
  ViennaCLInvalidTranspose, // for catching uninitialized and invalid values
  ViennaCLNoTrans,
  ViennaCLTrans
} ViennaCLTranspose;

typedef enum
{
  ViennaCLInvalidUplo, // for catching uninitialized and invalid values
  ViennaCLUpper,
  ViennaCLLower
} ViennaCLUplo;

typedef enum
{
  ViennaCLInvalidDiag, // for catching uninitialized and invalid values
  ViennaCLUnit,
  ViennaCLNonUnit
} ViennaCLDiag;

typedef enum
{
  ViennaCLInvalidPrecision,  // for catching uninitialized and invalid values
  ViennaCLFloat,
  ViennaCLDouble
} ViennaCLPrecision;

// Error codes:
typedef enum
{
  ViennaCLSuccess = 0,
  ViennaCLGenericFailure
} ViennaCLStatus;


/************* Backend Management ******************/

/** @brief Generic backend for CUDA, OpenCL, host-based stuff */
struct ViennaCLBackend_impl;
typedef ViennaCLBackend_impl*   ViennaCLBackend;

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendCreate(ViennaCLBackend * backend);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendSetOpenCLContextID(ViennaCLBackend backend, ViennaCLInt context_id);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLBackendDestroy(ViennaCLBackend * backend);

/******** User Types **********/

struct ViennaCLHostScalar_impl;
typedef ViennaCLHostScalar_impl*    ViennaCLHostScalar;

struct ViennaCLScalar_impl;
typedef ViennaCLScalar_impl*        ViennaCLScalar;

struct ViennaCLVector_impl;
typedef ViennaCLVector_impl*        ViennaCLVector;

struct ViennaCLMatrix_impl;
typedef ViennaCLMatrix_impl*        ViennaCLMatrix;


/******************** BLAS Level 1 ***********************/

// IxASUM

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLiamax(ViennaCLInt *alpha, ViennaCLVector x);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                                             ViennaCLInt *alpha,
                                                             float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                                             ViennaCLInt *alpha,
                                                             double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                                               ViennaCLInt *alpha,
                                                               cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                                               ViennaCLInt *alpha,
                                                               cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostiSamax(ViennaCLBackend backend, ViennaCLInt n,
                                                             ViennaCLInt *alpha,
                                                             float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostiDamax(ViennaCLBackend backend, ViennaCLInt n,
                                                             ViennaCLInt *alpha,
                                                             double *x, ViennaCLInt offx, ViennaCLInt incx);


// xASUM

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLasum(ViennaCLHostScalar *alpha, ViennaCLVector x);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASasum(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADasum(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSasum(ViennaCLBackend backend, ViennaCLInt n,
                                                              float *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDasum(ViennaCLBackend backend, ViennaCLInt n,
                                                              double *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSasum(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDasum(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);



// xAXPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLaxpy(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                              float alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                              double alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDaxpy(ViennaCLBackend backend, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);


// xCOPY

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLcopy(ViennaCLVector x, ViennaCLVector y);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAScopy(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADcopy(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLScopy(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                   cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                   cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostScopy(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDcopy(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

// xDOT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLdot(ViennaCLHostScalar *alpha, ViennaCLVector x, ViennaCLVector y);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASdot(ViennaCLBackend backend, ViennaCLInt n,
                                                           float *alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADdot(ViennaCLBackend backend, ViennaCLInt n,
                                                           double *alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSdot(ViennaCLBackend backend, ViennaCLInt n,
                                                             float *alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDdot(ViennaCLBackend backend, ViennaCLInt n,
                                                             double *alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSdot(ViennaCLBackend backend, ViennaCLInt n,
                                                           float *alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDdot(ViennaCLBackend backend, ViennaCLInt n,
                                                           double *alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy);

// xNRM2

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLnrm2(ViennaCLHostScalar *alpha, ViennaCLVector x);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                              float *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                              double *alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDnrm2(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);


// xROT

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLrot(ViennaCLVector     x,     ViennaCLVector y,
                                                      ViennaCLHostScalar c, ViennaCLHostScalar s);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASrot(ViennaCLBackend backend, ViennaCLInt n,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float c, float s);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADrot(ViennaCLBackend backend, ViennaCLInt n,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double c, double s);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSrot(ViennaCLBackend backend, ViennaCLInt n,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             float c, float s);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDrot(ViennaCLBackend backend, ViennaCLInt n,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             double c, double s);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSrot(ViennaCLBackend backend, ViennaCLInt n,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float c, float s);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDrot(ViennaCLBackend backend, ViennaCLInt n,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double c, double s);



// xSCAL

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLscal(ViennaCLHostScalar alpha, ViennaCLVector x);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASscal(ViennaCLBackend backend, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADscal(ViennaCLBackend backend, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSscal(ViennaCLBackend backend, ViennaCLInt n,
                                                              float alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDscal(ViennaCLBackend backend, ViennaCLInt n,
                                                              double alpha,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSscal(ViennaCLBackend backend, ViennaCLInt n,
                                                            float alpha,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDscal(ViennaCLBackend backend, ViennaCLInt n,
                                                            double alpha,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);


// xSWAP

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLswap(ViennaCLVector x, ViennaCLVector y);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASswap(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADswap(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSswap(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDswap(ViennaCLBackend backend, ViennaCLInt n,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSswap(ViennaCLBackend backend, ViennaCLInt n,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDswap(ViennaCLBackend backend, ViennaCLInt n,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);



/******************** BLAS Level 2 ***********************/

// xGEMV: y <- alpha * Ax + beta * y

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLgemv(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLVector x, ViennaCLHostScalar beta, ViennaCLVector y);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASgemv(ViennaCLBackend backend,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float beta,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADgemv(ViennaCLBackend backend,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double beta,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSgemv(ViennaCLBackend backend,
                                                              ViennaCLOrder order, ViennaCLTranspose transA,
                                                              ViennaCLInt m, ViennaCLInt n, float alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              float beta,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDgemv(ViennaCLBackend backend,
                                                              ViennaCLOrder order, ViennaCLTranspose transA,
                                                              ViennaCLInt m, ViennaCLInt n, double alpha, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                              double beta,
                                                              cl_mem y, ViennaCLInt offy, ViennaCLInt incy);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSgemv(ViennaCLBackend backend,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float beta,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDgemv(ViennaCLBackend backend,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double beta,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy);

// xTRSV: Ax <- x

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLtrsv(ViennaCLMatrix A, ViennaCLVector x, ViennaCLUplo uplo);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAStrsv(ViennaCLBackend backend,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADtrsv(ViennaCLBackend backend,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLStrsv(ViennaCLBackend backend,
                                                              ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                              ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDtrsv(ViennaCLBackend backend,
                                                              ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                              ViennaCLInt n, cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem x, ViennaCLInt offx, ViennaCLInt incx);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostStrsv(ViennaCLBackend backend,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDtrsv(ViennaCLBackend backend,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx);


// xGER: A <- alpha * x * y + A

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLger(ViennaCLHostScalar alpha, ViennaCLVector x, ViennaCLVector y, ViennaCLMatrix A);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASger(ViennaCLBackend backend,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m, ViennaCLInt n,
                                                           float alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADger(ViennaCLBackend backend,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m,  ViennaCLInt n,
                                                           double alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSger(ViennaCLBackend backend,
                                                             ViennaCLOrder order,
                                                             ViennaCLInt m, ViennaCLInt n,
                                                             float alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDger(ViennaCLBackend backend,
                                                             ViennaCLOrder order,
                                                             ViennaCLInt m, ViennaCLInt n,
                                                             double alpha,
                                                             cl_mem x, ViennaCLInt offx, ViennaCLInt incx,
                                                             cl_mem y, ViennaCLInt offy, ViennaCLInt incy,
                                                             cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSger(ViennaCLBackend backend,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m, ViennaCLInt n,
                                                           float alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDger(ViennaCLBackend backend,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m, ViennaCLInt n,
                                                           double alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda);



/******************** BLAS Level 3 ***********************/

// xGEMM: C <- alpha * AB + beta * C

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLgemm(ViennaCLHostScalar alpha, ViennaCLMatrix A, ViennaCLMatrix B, ViennaCLHostScalar beta, ViennaCLMatrix C);

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            float alpha,
                                                            float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            float beta,
                                                            float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            double alpha,
                                                            double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            double beta,
                                                            double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);

#ifdef VIENNACL_WITH_OPENCL
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLSgemm(ViennaCLBackend backend,
                                                              ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                              ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                              ViennaCLOrder orderC,
                                                              ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                              float alpha,
                                                              cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                              float beta,
                                                              cl_mem C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLOpenCLDgemm(ViennaCLBackend backend,
                                                              ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                              ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                              ViennaCLOrder orderC,
                                                              ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                              double alpha,
                                                              cl_mem A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                              cl_mem B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                              double beta,
                                                              cl_mem C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
#endif

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            float alpha,
                                                            float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            float beta,
                                                            float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);
VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            double alpha,
                                                            double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            double beta,
                                                            double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc);

// xTRSM: Triangular solves with multiple right hand sides

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLtrsm(ViennaCLMatrix A, ViennaCLUplo uplo, ViennaCLDiag diag, ViennaCLMatrix B);

#ifdef __cplusplus
}
#endif


#endif
