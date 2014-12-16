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
package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

class NBSparkTestSuite extends FunSuite with MahoutSuite with DistributedSparkSuite with NBTestBase {

  test("Spark NB Aggregator") {

    val rowBindings = new java.util.HashMap[String,Integer]()
    rowBindings.put("/Cat1/doc_a/", 0)
    rowBindings.put("/Cat2/doc_b/", 1)
    rowBindings.put("/Cat1/doc_c/", 2)
    rowBindings.put("/Cat2/doc_d/", 3)
    rowBindings.put("/Cat1/doc_e/", 4)


    val matrixSetup = sparse(
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil,
      (0, 0.0) ::(1, 0.1) ::(2, 0.0) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil,
      (0, 0.0) ::(1, 0.1) ::(2, 0.0) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil
    )


    matrixSetup.setRowLabelBindings(rowBindings)

    val TFIDFDrm = drm.drmParallelizeWithRowLabels(m = matrixSetup, numPartitions = 2)

    val (dslLabelIndex, dslAggregatedTFIDFDrm) = NaiveBayes.extractLabelsAndAggregateObservations(TFIDFDrm)
    val (sparkLabelIndex, sparkAggregatedTFIDFDrm) = SparkNaiveBayes.extractLabelsAndAggregateObservations(TFIDFDrm)

    dslLabelIndex.size should be (2)
    sparkLabelIndex.size should be (2)

    val dslCat1=dslLabelIndex("Cat1")
    val dslCat2=dslLabelIndex("Cat2")

    val sparkCat1=sparkLabelIndex("Cat1")
    val sparkCat2=sparkLabelIndex("Cat2")


    dslCat1 should be (0)
    dslCat2 should be (1)

    sparkCat1 should be (0)
    sparkCat2 should be (1)

    val dslAggInCore = dslAggregatedTFIDFDrm.collect
    val sparkAggInCore = sparkAggregatedTFIDFDrm.collect

    dslAggInCore.numCols should be (4) //4
    dslAggInCore.numRows should be (2) //2

    dslAggInCore(dslCat1, 0) - sparkAggInCore(dslCat1, 0) should be < epsilon //0.3
    dslAggInCore(dslCat1, 1) - sparkAggInCore(dslCat1, 1) should be < epsilon //0.0
    dslAggInCore(dslCat1, 2) - sparkAggInCore(dslCat1, 2) should be < epsilon //0.3
    dslAggInCore(dslCat1, 3) - sparkAggInCore(dslCat1, 3) should be < epsilon //0.0
    dslAggInCore(dslCat2, 0) - sparkAggInCore(dslCat2, 0) should be < epsilon //0.0
    dslAggInCore(dslCat2, 1) - sparkAggInCore(dslCat2, 1) should be < epsilon //0.2
    dslAggInCore(dslCat2, 2) - sparkAggInCore(dslCat2, 2) should be < epsilon //0.0
    dslAggInCore(dslCat2, 3) - sparkAggInCore(dslCat2, 3) should be < epsilon //0.2

  }
}
