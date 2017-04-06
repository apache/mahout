package org.apache.mahout.cuda

import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import scalabindings.RLikeOps._

import scala.util.Random

/**
  * Created by andy on 3/29/17.
  */
class CUDATestSuite extends FunSuite with Matchers {


  def getAverageTime(m: Int = 1000,
                     s: Int = 1000,
                     n: Int = 1000,
                     density: Double = .2,
                     seed: Long = 1234L,
                     nruns: Int = 5): Long = {


    val r = new Random(seed)
    val cudaCtx = new Context()

    // sparse row-wise
    val mxA = new SparseRowMatrix(m, s, false)
    val mxB = new SparseRowMatrix(s, n, true)

    // add some sparse data with the given threshold
    mxA := { (_, _, v) => if (r.nextDouble() < density) r.nextDouble() else v }
    mxB := { (_, _, v) => if (r.nextDouble() < density) r.nextDouble() else v }

    // run Mahout JVM - only math once
    var mxC = mxA %*% mxB

    // run Mahout JVM - only math another {{nruns}} times and take average
    var ms = System.currentTimeMillis()
    for (i: Int  <- 1 to nruns) {
      mxC = mxA %*% mxB
    }
    ms = (System.currentTimeMillis() - ms) / nruns
    info(s"Mahout JVM Sparse multiplication time: $ms ms.")


    // run Mahout JCuda math bindings once
    val cudaA = toCudaCmpMatrix(mxA, cudaCtx)
    val cudaB = toCudaCmpMatrix(mxB, cudaCtx)
    var mxCuda = prod(cudaA, cudaB, cudaCtx)

    // run Mahout JCuda another {{nruns}} times and take average
    ms = System.currentTimeMillis()
    for (i: Int  <- 1 to nruns) {
      mxCuda = prod(cudaA, cudaB, cudaCtx)
    }

    ms = (System.currentTimeMillis() - ms) / nruns
    info(s"Mahout JCuda Sparse multiplication time: $ms ms.")

    // TODO: Ensure that we've been working with the same matrices.
    // (mxC - mxCuda).norm / mxC.nrow / mxC.ncol should be < 1e-16
    ms
  }


  test("sparse mmul at geometry of 1000 x 1000 %*% 1000 x 1000 density = .2.  5 runs") {
     getAverageTime(1000, 1000, 1000, .20, 1234L, 5)
  }
  test("sparse mmul at geometry of 1000 x 1000 %*% 1000 x 1000 density = .02.  5 runs") {
    getAverageTime(1000, 1000, 1000, .02, 1234L, 5)
  }
  test("sparse mmul at geometry of 1000 x 1000 %*% 1000 x 1000 density = .002.  5 runs") {
    getAverageTime(1000, 1000, 1000, .002, 1234L, 5)
  }
}
