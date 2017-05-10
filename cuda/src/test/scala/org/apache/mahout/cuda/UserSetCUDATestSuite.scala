package org.apache.mahout.cuda

import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import scalabindings.RLikeOps._
import CUDATestSuite._
import scala.util.Properties.envOrElse

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
  m = envOrElse("SIZE_M","1000").toInt
  s = envOrElse("SIZE_S","1000").toInt
  n = envOrElse("SIZE_N","1000").toInt
  density = envOrElse("DENSITY",".02").toDouble
  seed = envOrElse("SEED","1234").toLong
  num_runs = envOrElse("NUM_RUNS","3").toInt

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
    for (i: Int <- 1 to nruns) {
      mxCuda = prod(cudaA, cudaB, cudaCtx)
    }

    ms = (System.currentTimeMillis() - ms) / nruns
    print(s"Mahout JCuda Sparse multiplication time: $ms ms.")

    // TODO: Ensure that we've been working with the same matrices.
//     (mxC - fromCudaCmpMatrix(mxCuda)).norm / mxC.nrow / mxC.ncol should be < 1e-16
     assert(((Math.abs((mxC - fromCudaCmpMatrix(mxCuda)).norm / mxC.nrow / mxC.ncol)) - 1e-16) < 0)
    ms
  }

}

