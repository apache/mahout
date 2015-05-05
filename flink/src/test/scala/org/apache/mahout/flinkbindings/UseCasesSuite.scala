package org.apache.mahout.flinkbindings

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import org.apache.mahout.flinkbindings._
import org.apache.mahout.math.function.IntIntFunction
import scala.util.Random
import scala.util.MurmurHash
import scala.util.hashing.MurmurHash3
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.scalatest.Ignore

@RunWith(classOf[JUnitRunner])
class UseCasesSuite extends FunSuite with DistributedFlinkSuit {

  val LOGGER = LoggerFactory.getLogger(getClass())

  test("use case: Power interation 1000 x 1000 matrix") {
    val dim = 1000

    // we want a symmetric matrix so we can have real eigenvalues
    val inCoreA = symmtericMatrix(dim, max = 2000)

    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    var x: Vector = 1 to dim map (_ => 1.0 / Math.sqrt(dim))
    var converged = false

    var iteration = 1

    while (!converged) {
      LOGGER.info(s"iteration #$iteration...")

      val Ax = A %*% x
      var x_new = Ax.collect(::, 0)
      x_new = x_new / x_new.norm(2)

      val diff = (x_new - x).norm(2)
      LOGGER.info(s"difference norm is $diff")

      converged = diff < 1e-6
      iteration = iteration + 1
      x = x_new
    }

    LOGGER.info("converged")
    // TODO: add test that it's the 1st PC
  }

  def symmtericMatrix(dim: Int, max: Int, seed: Int = 0x31337) = {
    Matrices.functionalMatrixView(dim, dim, new IntIntFunction {
      def apply(i: Int, j: Int): Double = {
        val arr = Array(i + j, i * j, i + j + 31, i / (j + 1) + j / (i + 1))
        Math.abs(MurmurHash3.arrayHash(arr, seed) % max)
      }
    })
  }

  test("use case: OLS Regression") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4), (5, 6), (7, 8), (9, 10))
    val x = dvec(1, 2, 2, 3, 3, 3)
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val AtA = A.t %*% A
    val Atx = A.t %*% x

    val w = solve(AtA, Atx)

    val expected = solve(inCoreA.t %*% inCoreA, inCoreA.t %*% x)
    assert((w(::, 0) - expected).norm(2) < 1e-6)
  }

}