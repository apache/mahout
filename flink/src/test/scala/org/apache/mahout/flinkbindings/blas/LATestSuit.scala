package org.apache.mahout.flinkbindings.blas

import org.scalatest.FunSuite
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import org.apache.mahout.flinkbindings._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.apache.mahout.math.drm.logical.OpAx
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.drm.logical._

@RunWith(classOf[JUnitRunner])
class LATestSuit extends FunSuite with DistributedFlinkSuit {

  test("Ax blockified") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x: Vector = (0, 1, 2)

    val opAx = new OpAx(A, x)
    val res = FlinkOpAx.blockifiedBroadcastAx(opAx, A)
    val drm = new CheckpointedFlinkDrm(res.deblockify.ds)
    val output = drm.collect

    val b = output(::, 0)
    assert(b == dvec(8, 11, 14))
  }

  test("At sparseTrick") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val opAt = new OpAt(A)
    val res = FlinkOpAt.sparseTrick(opAt, A)
    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreA.ncol, _ncol=inCoreA.nrow)
    val output = drm.collect

    assert((output - inCoreA.t).norm < 1e-6)
  }

  test("AtB notZippable") {
    val inCoreAt = dense((1, 2), (2, 3), (3, 4))

    val At = drmParallelize(m = inCoreAt, numPartitions = 2)

    val inCoreB = dense((1, 2), (3, 4), (11, 4))
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val opAtB = new OpAtB(At, B)
    val res = FlinkOpAtB.notZippable(opAtB, At, B)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreAt.ncol, _ncol=inCoreB.ncol)
    val output = drm.collect

    val expected = inCoreAt.t %*% inCoreB
    assert((output - expected).norm < 1e-6)
  }

  test("AewScalar opScalarNoSideEffect") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val scalar = 5.0

    val op = new OpAewScalar(A, scalar, "*") 
    val res = FlinkOpAewScalar.opScalarNoSideEffect(op, A, scalar)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreA.nrow, _ncol=inCoreA.ncol)
    val output = drm.collect

    val expected = inCoreA  * scalar
    assert((output - expected).norm < 1e-6)
  }

  test("AewB rowWiseJoinNoSideEffect") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val op = new OpAewB(A, A, "*")
    val res = FlinkOpAewB.rowWiseJoinNoSideEffect(op, A, A)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreA.nrow, _ncol=inCoreA.ncol)
    val output = drm.collect

    assert((output - (inCoreA  * inCoreA)).norm < 1e-6)
  }

  test("Cbind") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((4, 4), (5, 5), (6, 7))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val op = new OpCbind(A, B)
    val res = FlinkOpCBind.cbind(op, A, B)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreA.nrow, 
        _ncol=(inCoreA.ncol + inCoreB.ncol))
    val output = drm.collect

    val expected = dense((1, 2, 4, 4), (2, 3, 5, 5), (3, 4, 6, 7))
    assert((output - expected).norm < 1e-6)
  }

  test("slice") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 7))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val range = 2 until 5
    val op = new OpRowRange(A, range)
    val res = FlinkOpRowRange.slice(op, A)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=op.nrow, 
        _ncol=inCoreA.ncol)
    val output = drm.collect

    val expected = inCoreA(2 until 5, ::)
    assert((output - expected).norm < 1e-6)
  }

  test("A times inCoreB") {
    val inCoreA = dense((1, 2, 3), (2, 3, 1), (3, 4, 4), (4, 4, 5), (5, 5, 7), (6, 7, 11))
    val inCoreB = dense((2, 1), (3, 4), (5, 11))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val op = new OpTimesRightMatrix(A, inCoreB)
    val res = FlinkOpTimesRightMatrix.drmTimesInCore(op, A, inCoreB)

    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=op.nrow, 
        _ncol=op.ncol)
    val output = drm.collect

    val expected = inCoreA %*% inCoreB
    assert((output - expected).norm < 1e-6)
  }

}