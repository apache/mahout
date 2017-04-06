package org.apache.mahout.cuda

import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import scalabindings.RLikeOps._
import CUDATestSuite._

import scala.util.Random


import scala.util.Random
/**
  * Created by andy on 3/29/17.
  */

// some quickfixes as well
class UserSetCUDATestSuite extends FunSuite with Matchers {

  // defaults
  var m: Int = 1000
  var s: Int = 1000
  var n: Int = 1000
  var density: Double = .2
  var seed: Long = 1234L
  var num_runs: Int = 5

  // grab the environment variables if set.
  m = sys.env("SIZE_M").toInt
  s = sys.env("SIZE_S").toInt
  n = sys.env("SIZE_N").toInt
  density = sys.env("DENSITY").toDouble
  seed = sys.env("SEED").toLong
  num_runs = sys.env("NUM_RUNS").toInt

  test("User Defined sparse mmul at geometry of "
    + m + " x " + s + " %*% " + s + " x " + n + " density = " + density + " " +  num_runs + " runs \n") {

   val ms =  getAverageTime(m, n, s, density, seed, num_runs)

    println("User Defined sparse mmul at geometry of "
      + m + " x " + s + " %*% " + s + " x " + n + " density = " + density + " " + num_runs + " runs : "+ms +" ms")
  }
}


object UserSetCUDATestSuite {
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
    print(s"Mahout JVM Sparse multiplication time: $ms ms.")


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
    print(s"Mahout JCuda Sparse multiplication time: $ms ms.")

    // TODO: Ensure that we've been working with the same matrices.
    // (mxC - mxCuda).norm / mxC.nrow / mxC.ncol should be < 1e-16
    ms
  }

}

