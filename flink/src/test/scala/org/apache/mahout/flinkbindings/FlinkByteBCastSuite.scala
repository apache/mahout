package org.apache.mahout.flinkbindings

import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.scalatest.FunSuite

class FlinkByteBCastSuite extends FunSuite {

  test("BCast vector") {
    val v = dvec(1, 2, 3)
    val vBc = FlinkByteBCast.wrap(v)
    assert((v - vBc.value).norm(2) <= 1e-6)
  }

  test("BCast matrix") {
    val m = dense((1, 2), (3, 4))
    val mBc = FlinkByteBCast.wrap(m)
    assert((m - mBc.value).norm <= 1e-6)
  }
}