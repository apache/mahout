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
import org.apache.mahout.math.drm.logical.OpAt

@RunWith(classOf[JUnitRunner])
class LATestSuit extends FunSuite with DistributedFlinkSuit {

  test("Ax") {
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

  test("At") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val opAt = new OpAt(A)
    val res = FlinkOpAt.sparseTrick(opAt, A)
    val drm = new CheckpointedFlinkDrm(res.deblockify.ds, _nrow=inCoreA.ncol, _ncol=inCoreA.nrow)
    val output = drm.collect

    assert((output - inCoreA.t).norm < 1e-6)
  }

}