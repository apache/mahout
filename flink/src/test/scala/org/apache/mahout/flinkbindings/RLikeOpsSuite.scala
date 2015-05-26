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
class RLikeOpsSuite extends FunSuite with DistributedFlinkSuit {

  val LOGGER = LoggerFactory.getLogger(getClass())

  ignore("A %*% x") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x: Vector = (0, 1, 2)

    val res = A %*% x

    val b = res.collect(::, 0)
    assert(b == dvec(8, 11, 14))
  }

  ignore("A.t") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val res = A.t.collect

    val expected = inCoreA.t
    assert((res - expected).norm < 1e-6)
  }

  ignore("A.t %*% x") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x = dvec(3, 11)
    val res = (A.t %*% x).collect(::, 0)

    val expected = inCoreA.t %*% x 
    assert((res - expected).norm(2) < 1e-6)
  }

  ignore("A.t %*% B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A.t %*% B

    val expected = inCoreA.t %*% inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A %*% B.t") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A %*% B.t

    val expected = inCoreA %*% inCoreB.t
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A.t %*% A") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A.t %*% A

    val expected = inCoreA.t %*% inCoreA
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A %*% B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4)).t
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A %*% B

    val expected = inCoreA %*% inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A * scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A * 5
    assert((res.collect - inCoreA * 5).norm < 1e-6)
  }

  ignore("A / scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4)).t
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A / 5
    assert((res.collect - (inCoreA / 5)).norm < 1e-6)
  }

  ignore("A + scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A + 5
    assert((res.collect - (inCoreA + 5)).norm < 1e-6)
  }

  ignore("A - scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A - 5
    assert((res.collect - (inCoreA - 5)).norm < 1e-6)
  }

  ignore("A * B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A * B
    val expected = inCoreA * inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A / B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A / B
    val expected = inCoreA / inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A + B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A + B
    val expected = inCoreA + inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A - B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A - B
    val expected = inCoreA - inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A cbind B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A cbind B
    val expected = dense((1, 2, 1, 2), (2, 3, 3, 4), (3, 4, 11, 4))
    assert((res.collect - expected).norm < 1e-6)
  }

  ignore("A rbind B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A rbind B
    val expected = dense((1, 2), (2, 3), (3, 4), (1, 2), (3, 4), (11, 4))
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A row slice") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 7))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A(2 until 5, ::)
    val expected = inCoreA(2 until 5, ::)
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A column slice") {
    val inCoreA = dense((1, 2, 1, 2), (2, 3, 3, 4), (3, 4, 11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A(::, 0 until 2)
    val expected = inCoreA(::, 0 until 2)
    assert((res.collect - expected).norm < 1e-6)
  }

}