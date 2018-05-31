/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.sparkbindings.drm

import org.scalatest.FunSuite
import org.apache.mahout.math._
import scalabindings._
import drm._
import RLikeOps._
import RLikeDrmOps._
import org.apache.mahout.logging.info
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite

case class Thingy(thing1: Double, thing2: Double, thing3: Double)

/** DRMLike tests -- just run common DRM tests in Spark. */
class DrmLikeSuite extends FunSuite with DistributedSparkSuite with DrmLikeSuiteBase {

  test("drmParallellize produces drm with no missing rows") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.canHaveMissingRows shouldBe false
  }

  test("DRM blockify dense") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (inCoreA - drmA.mapBlock() {
      case (keys, block) =>
        if (!block.isInstanceOf[DenseMatrix])
          throw new AssertionError("Block must be dense.")
        keys -> block
    }).norm should be < 1e-4
  }

  test("DRM blockify sparse -> DRM") {

    val inCoreA = sparse(
      (1, 2, 3),
      0 -> 3 :: 2 -> 5 :: Nil
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (inCoreA - drmA.mapBlock() {
      case (keys, block) =>
        if (block.isInstanceOf[SparseRowMatrix])
          throw new AssertionError("Block must be dense.")
        keys -> block
    }).norm should be < 1e-4

  }

  test("DRM wrap labeled points") {

    import org.apache.spark.mllib.linalg.{Vectors => SparkVector}
    import org.apache.spark.mllib.regression.LabeledPoint

    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    val lpRDD = sc.parallelize(Seq(LabeledPoint(1.0, SparkVector.dense(2.0, 0.0, 4.0)),
                                   LabeledPoint(2.0, SparkVector.dense(3.0, 0.0, 5.0)),
                                   LabeledPoint(3.0, SparkVector.dense(4.0, 0.0, 6.0)) ))

    val lpDRM = drmWrapMLLibLabeledPoint(rdd = lpRDD)
    val lpM = lpDRM.collect(::,::)
    val testM = dense((2,0,4,1), (3,0,5,2), (4,0,6,3))
    assert(lpM === testM)
  }

  test("DRM wrap spark vectors") {

    import org.apache.spark.mllib.linalg.{Vectors => SparkVector}

    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    val svRDD = sc.parallelize(Seq(SparkVector.dense(2.0, 0.0, 4.0),
                                   SparkVector.dense(3.0, 0.0, 5.0),
                                   SparkVector.dense(4.0, 0.0, 6.0) ))

    val svDRM = drmWrapMLLibVector(rdd = svRDD)
    val svM = svDRM.collect(::,::)
    val testM = dense((2,0,4), (3,0,5), (4,0,6))

    assert(svM === testM)

    val ssvRDD = sc.parallelize(Seq(SparkVector.sparse(3, Array(1,2), Array(3,4)),
      SparkVector.sparse(3, Array(0,2), Array(3,4)),
      SparkVector.sparse(3, Array(0,1), Array(3,4))) )

    val ssvDRM = drmWrapMLLibVector(rdd = ssvRDD)
    val ssvM = ssvDRM.collect(::,::)

    val testSM = sparse(
      (1, 3) :: (2, 4) :: Nil,
      (0, 3) :: (2, 4) :: Nil,
      (0, 3) :: (1, 4) :: Nil)

    assert(ssvM === testSM)
  }



  test("DRM wrap spark dataframe") {

    import org.apache.spark.mllib.linalg.{Vectors => SparkVector}

    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val myDF = sc.parallelize(Seq((2.0, 0.0, 4.0),
                                  (3.0, 0.0, 5.0),
                                  (4.0, 0.0, 6.0) ))
                    .map(o => Thingy(o._1, o._2, o._3))
                    .toDF()

    val dfDRM = drmWrapDataFrame(df = myDF)
    val dfM = dfDRM.collect(::,::)
    val testM = dense((2,0,4), (3,0,5), (4,0,6))

    assert(dfM === testM)
  }

  test("Aggregating transpose") {

    val mxA = new DenseMatrix(20, 10) := 1

    val drmA = drmParallelize(mxA, numPartitions = 3)

    val reassignedA = drmA.mapBlock() { case (keys, block) ⇒
      keys.map(_ % 3) → block
    }

    val mxAggrA = reassignedA.t(::, 0 until 3).collect

    info(mxAggrA.toString)

    mxAggrA(0,0) shouldBe 7
    mxAggrA(0,1) shouldBe 7
    mxAggrA(0,2) shouldBe 6
  }
}
