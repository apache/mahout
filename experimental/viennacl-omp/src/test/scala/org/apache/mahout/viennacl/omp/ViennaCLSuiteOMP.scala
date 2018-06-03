/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.viennacl.openmp

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.bytedeco.javacpp.DoublePointer
import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.viennacl.openmp.javacpp._
import org.apache.mahout.viennacl.openmp.javacpp.Functions._
import org.apache.mahout.viennacl.openmp.javacpp.LinalgFunctions._

import scala.util.Random

class ViennaCLSuiteOMP extends FunSuite with Matchers {

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
    val hostClCtx = new Context(Context.MAIN_MEMORY)
    val cpuMx = new DenseRowMatrix(data = data, nrow = m, ncol = n, hostClCtx)
    // And free.
    cpuMx.close()

  }


  test("mmul microbenchmark") {
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

  }

  test("trans") {

    val ompCtx = new Context(Context.MAIN_MEMORY)


    val m = 20
    val n = 30

    val r = new Random(1234)

    // Dense row-wise
    val mxA = new DenseMatrix(m, n)

    // add some data
    mxA := { (_, _, _) => r.nextDouble() }


    // Test transposition in OpenMP
    {
      val ompA = toVclDenseRM(src = mxA, ompCtx)
      val ompAt = new DenseRowMatrix(trans(ompA))

      val mxAt = fromVclDenseRM(ompAt)
      ompA.close()
      ompAt.close()

      (mxAt - mxA.t).norm / m / n should be < 1e-16
    }

  }

  test("sparse mmul microbenchmark") {

    val ompCtx = new Context(Context.MAIN_MEMORY)

    val m = 3000
    val n = 3000
    val s = 1000

    val r = new Random(1234)

    // sparse row-wise
    val mxA = new SparseRowMatrix(m, s, false)
    val mxB = new SparseRowMatrix(s, n, true)

    // add some sparse data with 20% density
    mxA := { (_, _, v) => if (r.nextDouble() < .20) r.nextDouble() else v }
    mxB := { (_, _, v) => if (r.nextDouble() < .20) r.nextDouble() else v }

    var ms = System.currentTimeMillis()
    val mxC = mxA %*% mxB
    ms = System.currentTimeMillis() - ms
    info(s"Mahout Sparse multiplication time: $ms ms.")


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
      (mxC - ompMxC).norm / mxC.nrow / mxC.ncol should be < 1e-10

      ompA.close()
      ompB.close()
      ompC.close()

    }

  }

  test("VCL Dense Matrix %*% Dense vector - no OpenCl") {

    val ompCtx = new Context(Context.MAIN_MEMORY)


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


    //Test multiplication in OpenMP
      {

        ms = System.currentTimeMillis()
        val ompMxA = toVclDenseRM(mxA, ompCtx)
        val ompVecB = toVclVec(dvecB, ompCtx)

        val ompVecC = new VCLVector(prod(ompMxA, ompVecB))
        val ompDvecC = fromVClVec(ompVecC)

        ms = System.currentTimeMillis() - ms
        info(s"ViennaCL/cpu/OpenMP dense matrix %*% dense vector multiplication time: $ms ms.")
        (ompDvecC.toColMatrix - mDvecC.toColMatrix).norm / s  should be < 1e-10

        ompMxA.close()
        ompVecB.close()
        ompVecC.close()
      }

  }


  test("Sparse %*% Dense mmul microbenchmark") {
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

  }



}
