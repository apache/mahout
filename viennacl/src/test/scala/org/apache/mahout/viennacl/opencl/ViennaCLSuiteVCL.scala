/**
  * Licensed to the Apache Software Foundation (ASF) under one or more
  * contributor license agreements.  See the NOTICE file distributed with
  * this work for additional information regarding copyright ownership.
  * The ASF licenses this file to You under the Apache License, Version 2.0
  * (the "License"); you may not use this file except in compliance with
  * the License.  You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
package org.apache.mahout.opencl.viennacl

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.viennacl.opencl.javacpp.CompressedMatrix
import org.apache.mahout.viennacl.opencl._
import org.apache.mahout.viennacl.opencl.javacpp.Functions._
import org.apache.mahout.viennacl.opencl.javacpp.LinalgFunctions._
import org.apache.mahout.viennacl.opencl.javacpp.{Context, LinalgFunctions, VCLVector, _}
import org.bytedeco.javacpp.DoublePointer
import org.scalatest.{FunSuite, Matchers}

import scala.util.Random

class ViennaCLSuiteVCL extends FunSuite with Matchers {

  test("row-major viennacl::matrix") {

    // Just to make sure the javacpp library is loaded:
    Context.loadLib()

    val m = 20
    val n = 30
    val data = new DoublePointer(m * n)
    val buff = data.asBuffer()
    // Fill with some noise
    while (buff.remaining() > 0) buff.put(Random.nextDouble())

    // Create row-major matrix with OpenCL
    val openClCtx = new Context(Context.OPENCL_MEMORY)
    val hostClCtx = new Context(Context.MAIN_MEMORY)
    val oclMx = new DenseRowMatrix(m, n, openClCtx)
    val cpuMx = new DenseRowMatrix(data = data, nrow = m, ncol = n, hostClCtx)

    oclMx.memoryDomain shouldBe Context.OPENCL_MEMORY

    // Apparently, this doesn't really switch any contexts? any how, uncommenting this causes
    // subsequent out-of-resources OCL error for me in other tests. Perhaps we shouldnt' really
    // do cross-memory-domain assigns?

    //    oclMx := cpuMx

    // Did it change memory domain? that may explain the OCL resource leak.
    info(s"OCL matrix memory domain after assgn=${oclMx.memoryDomain}")
    oclMx.memoryDomain shouldBe Context.OPENCL_MEMORY


    // And free.
    cpuMx.close()
    oclMx.close()

  }

  test("dense vcl mmul with fast_copy") {

    import LinalgFunctions._

    val vclCtx = new Context(Context.OPENCL_MEMORY)

    val m = 20
    val n = 30
    val s = 40

    val r = new Random(1234)

    // Dense row-wise
    val mxA = new DenseMatrix(m, s)
    val mxB = new DenseMatrix(s, n)

    // add some data
    mxA := { (_, _, _) => r.nextDouble() }
    mxB := { (_, _, _) => r.nextDouble() }

    // time Mahout MMul
    // mxC = mxA %*% mxB via Mahout MMul
    val mxCControl = mxA %*% mxB

    val vclA = toVclDenseRM(mxA, vclCtx)
    val vclB = toVclDenseRM(mxB, vclCtx)

    val vclC = new DenseRowMatrix(prod(vclA, vclB))

    val mxC = fromVclDenseRM(vclC)

    vclA.close()
    vclB.close()
    vclC.close()

    // So did we compute it correctly?
    (mxC - mxA %*% mxB).norm / m / n should be < 1e-16

    vclCtx.deallocate()
    vclCtx.close()

  }

  test("mmul microbenchmark") {
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val memCtx = new Context(Context.MAIN_MEMORY)

    val m = 3000
    val n = 3000
    val s = 1000

    val r = new Random(1234)

    // Dense row-wise
    val mxA = new DenseMatrix(m, s)
    val mxB = new DenseMatrix(s, n)

    // add some data
    mxA := { (_, _, _) => r.nextDouble() }
    mxB := { (_, _, _) => r.nextDouble() }

    var ms = System.currentTimeMillis()
    mxA %*% mxB
    ms = System.currentTimeMillis() - ms
    info(s"Mahout multiplication time: $ms ms.")

    import LinalgFunctions._

    // openCL time, including copying:
    {
      ms = System.currentTimeMillis()
      val oclA = toVclDenseRM(mxA, oclCtx)
      val oclB = toVclDenseRM(mxB, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()
    }

    // openMP/cpu time, including copying:
    {
      ms = System.currentTimeMillis()
      val ompA = toVclDenseRM(mxA, memCtx)
      val ompB = toVclDenseRM(mxB, memCtx)
      val ompC = new DenseRowMatrix(prod(ompA, ompB))
      val mxC = fromVclDenseRM(ompC)
      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/cpu/OpenMP multiplication time: $ms ms.")

      ompA.close()
      ompB.close()
      ompC.close()
    }
    oclCtx.deallocate()
    oclCtx.close()


  }

  test("trans") {

    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val ompCtx = new Context(Context.MAIN_MEMORY)


    val m = 20
    val n = 30

    val r = new Random(1234)

    // Dense row-wise
    val mxA = new DenseMatrix(m, n)

    // add some data
    mxA := { (_, _, _) => r.nextDouble() }

    // Test transposition in OpenCL
    {
      val oclA = toVclDenseRM(src = mxA, oclCtx)
      val oclAt = new DenseRowMatrix(trans(oclA))

      val mxAt = fromVclDenseRM(oclAt)
      oclA.close()
      oclAt.close()

      (mxAt - mxA.t).norm / m / n should be < 1e-16
    }

    // Test transposition in OpenMP
    {
      val ompA = toVclDenseRM(src = mxA, ompCtx)
      val ompAt = new DenseRowMatrix(trans(ompA))

      val mxAt = fromVclDenseRM(ompAt)
      ompA.close()
      ompAt.close()

      (mxAt - mxA.t).norm / m / n should be < 1e-16
    }
    oclCtx.deallocate()
    oclCtx.close()


  }

  test("sparse mmul microbenchmark") {

    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val ompCtx = new Context(Context.MAIN_MEMORY)


    val m = 3000
    val n = 3000
    val s = 1000

    val r = new Random(1234)

    // sparse row-wise
    val mxA = new SparseRowMatrix(m, s, false)
    val mxB = new SparseRowMatrix(s, n, true)

    // add some sparse data with a 20% threshold
    mxA := { (_, _, v) => if (r.nextDouble() < .20) r.nextDouble() else v }
    mxB := { (_, _, v) => if (r.nextDouble() < .20) r.nextDouble() else v }

    var ms = System.currentTimeMillis()
    val mxC = mxA %*% mxB
    ms = System.currentTimeMillis() - ms
    info(s"Mahout Sparse multiplication time: $ms ms.")

//     Test multiplication in OpenCL
    {

      ms = System.currentTimeMillis()
      val oclA = toVclCmpMatrixAlt(mxA, oclCtx)
      val oclB = toVclCmpMatrixAlt(mxB, oclCtx)

      val oclC = new CompressedMatrix(prod(oclA, oclB))
      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/OpenCL Sparse multiplication time: $ms ms.")

      val oclMxC = fromVclCompressedMatrix(oclC)
      val ompMxC = fromVclCompressedMatrix(oclC)
      (mxC - oclMxC).norm / mxC.nrow / mxC.ncol should be < 1e-16

      oclA.close()
      oclB.close()
      oclC.close()
    }

    // Test multiplication in OpenMP
    {
      ms = System.currentTimeMillis()
      //      val ompA = toVclCompressedMatrix(src = mxA, ompCtx)
      //      val ompB = toVclCompressedMatrix(src = mxB, ompCtx)

      val ompA = toVclCmpMatrixAlt(mxA, ompCtx)
      val ompB = toVclCmpMatrixAlt(mxB, ompCtx)

      val ompC = new CompressedMatrix(prod(ompA, ompB))

      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/cpu/OpenMP Sparse multiplication time: $ms ms.")

      val ompMxC = fromVclCompressedMatrix(ompC)
      (mxC - ompMxC).norm / mxC.nrow / mxC.ncol should be < 1e-16

      ompA.close()
      ompB.close()
      ompC.close()

    }
    oclCtx.deallocate()
    oclCtx.close()

  }

  test("VCL Dense Matrix %*% Dense vector") {

    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val mainCtx = new Context(Context.MAIN_MEMORY)


    val m = 3000
    val s = 1000

    val r = new Random(1234)

    // Dense row-wise
    val mxA = new DenseMatrix(m, s)
    val dvecB = new DenseVector(s)

    // add some random data
    mxA := { (_,_,_) => r.nextDouble() }
    dvecB := { (_,_) => r.nextDouble() }

    //test in matrix %*% vec
    var ms = System.currentTimeMillis()
    val mDvecC = mxA %*% dvecB
    ms = System.currentTimeMillis() - ms
    info(s"Mahout dense matrix %*% dense vector multiplication time: $ms ms.")


    // Test mx %*% vec multiplication in OpenCL
      {

        ms = System.currentTimeMillis()

        // we must first create vectors in main memory
        // when working with vectors at least in ViennaCl
        // this is the preferred method
        val oclMxA = toVclDenseRM(mxA, mainCtx)
        val oclVecB = toVclVec(dvecB, mainCtx)

        // now copy to the OpenCL device
        oclMxA.switchMemoryContext(oclCtx)
        oclVecB.switchMemoryContext(oclCtx)

        // perform multiplication
        val oclVecC = new VCLVector(prod(oclMxA, oclVecB))

        // copy back to main memory so that we may
        // read values out of the result. This must be
        // copied back to main memory VCL can not read
        // directly from an OpenCL device
        oclVecC.switchMemoryContext(mainCtx)

        val oclDvecC = fromVClVec(oclVecC)

        ms = System.currentTimeMillis() - ms
        info(s"ViennaCL/OpenCL dense matrix %*% dense vector multiplication time: $ms ms.")
        (oclDvecC.toColMatrix - mDvecC.toColMatrix).norm / s should be < 1e-10

        oclMxA.close()
        oclVecB.close()
        oclVecC.close()
      }

      //Test multiplication in OpenMP
      {

        ms = System.currentTimeMillis()
        val ompMxA = toVclDenseRM(mxA, mainCtx)
        val ompVecB = toVclVec(dvecB, mainCtx)

        val ompVecC = new VCLVector(prod(ompMxA, ompVecB))
        val ompDvecC = fromVClVec(ompVecC)

        ms = System.currentTimeMillis() - ms
        info(s"ViennaCL/cpu/OpenMP dense matrix %*% dense vector multiplication time: $ms ms.")
        (ompDvecC.toColMatrix - mDvecC.toColMatrix).norm / s  should be < 1e-10

        ompMxA.close()
        ompVecB.close()
        ompVecC.close()
      }

      oclCtx.deallocate()
      oclCtx.close()

  }


  test("Sparse %*% Dense mmul microbenchmark") {
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val memCtx = new Context(Context.MAIN_MEMORY)

    val m = 3000
    val n = 3000
    val s = 1000

    val r = new Random(1234)

    // Dense row-wise
    val mxSr = new SparseMatrix(m, s)
    val mxDn = new DenseMatrix(s, n)

    // add some data
    mxSr := { (_, _, v) => if (r.nextDouble() < .20) r.nextDouble() else v }
    mxDn := { (_, _, _) => r.nextDouble() }

    var ms = System.currentTimeMillis()
    mxSr %*% mxDn
    ms = System.currentTimeMillis() - ms
    info(s"Mahout multiplication time: $ms ms.")

    import LinalgFunctions._

    // For now, since our dense matrix is fully dense lets just assume that our result is dense.
    // openCL time, including copying:
    {
      ms = System.currentTimeMillis()
      val oclA = toVclCmpMatrixAlt(mxSr, oclCtx)
      val oclB = toVclDenseRM(mxDn, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()
    }

    // openMP/cpu time, including copying:
    {
      ms = System.currentTimeMillis()
      val ompA = toVclCmpMatrixAlt(mxSr, memCtx)
      val ompB = toVclDenseRM(mxDn, memCtx)
      val ompC = new DenseRowMatrix(prod(ompA, ompB))
      val mxC = fromVclDenseRM(ompC)
      ms = System.currentTimeMillis() - ms
      info(s"ViennaCL/cpu/OpenMP multiplication time: $ms ms.")

      ompA.close()
      ompB.close()
      ompC.close()
    }

    oclCtx.deallocate()
    oclCtx.close()


  }



}
