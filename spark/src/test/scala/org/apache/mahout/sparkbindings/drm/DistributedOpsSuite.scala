package org.apache.mahout.sparkbindings.drm

import org.apache.mahout.math.{Vector, Matrix}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.FunSuite

/**
 * @author gokhan
 */
class DistributedOpsSuite extends FunSuite with DistributedSparkSuite{
  test("acummulateBlocks") {
    import RLikeDrmOps._

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))

    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)

    val accControl = drmA.zSum

    val seqOp: (Double, (Array[Int], _<: Matrix)) => Double = (value, matrix) =>
      value + matrix._2.zSum

    val combOp: (Double, Double) => Double = (val1, val2) => val1 + val2

    //check if it accumulates correctly
    val acc1 = drmA.accumulateBlocks(0.0, seqOp, combOp)

    //check if it can be the called after a series of matrix ops
    val drmB = drmA + 1
    val accControl2 = drmB.zSum()
    val acc2 = drmB.accumulateBlocks(0.0, seqOp, combOp)
    drmB.colSums()


    println(acc1)
    println(acc2)

    assert(acc1 == accControl)
    assert(acc2 == accControl2)
  }

  test("accumulateRows") {
    import RLikeDrmOps._
    import RLikeOps._
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))

    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)

    val accControl = drmA.zSum

    val seqOp: (Double, (Int, Vector)) => Double = (value, vector) =>
      value + vector._2.sum

    val combOp: (Double, Double) => Double = (val1, val2) => val1 + val2

    //check if it accumulates correctly
    val acc1 = drmA.accumulateRows(0.0, seqOp, combOp)

    //check if it can be the called after a series of matrix ops

    val drmB = drmA + 1
    val accControl2 = drmB.zSum()
    val acc2 = (drmA + 1).accumulateRows(0.0, seqOp, combOp)


    println(acc1)
    println(acc2)

    assert(acc1 == accControl)
    assert(acc2 == accControl2)
  }

}
