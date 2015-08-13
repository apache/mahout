package org.apache.mahout.flinkbindings.standard

import org.apache.mahout.flinkbindings._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.apache.mahout.math.decompositions.DistributedDecompositionsSuiteBase


@RunWith(classOf[JUnitRunner])
class DistributedDecompositionsSuite extends FunSuite with DistributedFlinkSuite
      with DistributedDecompositionsSuiteBase {

}