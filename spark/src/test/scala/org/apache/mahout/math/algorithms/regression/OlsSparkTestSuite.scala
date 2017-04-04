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

package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.regression._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite


class OlsSparkTestSuite extends FunSuite with MahoutSuite with DistributedSparkSuite with OrdinaryLeastSquaresTest {
  // Common tests located in OrdinaryLeastSquaresTest.scala
  // The test below is common to spark as I created an random RDD for larger size
  test("Simple Medium Model2 - Spark Specific") {

    val a = Range(0,10).toArray.map { a => Range(0,10).toArray }

    import org.apache.spark.mllib.util.LinearDataGenerator
    import org.apache.mahout.math._
    import org.apache.mahout.sparkbindings._

    val n = 10000
    val features = 100
    val eps = 10 // i'm guessing error term, poorly documented
    val partitions = 2
    val intercept = 10.0
    //val sc = new SparkContext()

    val synDataRDD = LinearDataGenerator.generateLinearRDD(mahoutCtx, n, features, eps, partitions, intercept)

    // Maybe refine this with something from the sparkbindings, but for now this is good...
    val tempRDD = synDataRDD.zipWithIndex.map( lv => {
      val K = lv._2.toInt
      val x = new DenseVector(lv._1.features.toArray )
      //x = sparkVec2mahoutVec( lv._1.features ) // still doesn't serialize
      val y = lv._1.label
      (K, (y, x))
    }).persist

    //println("----------- Creating DRMs -------------- ")

    // temp RDD to X an y
    val drmRddX : DrmRdd[Int] = tempRDD.map(o => (o._1, o._2._2))
    val drmX = drmWrap(rdd= drmRddX)
    val drmRddY:DrmRdd[Int] = tempRDD.map(o => (o._1, new DenseVector( Array(o._2._1) )))
    val drmy = drmWrap(rdd= drmRddY)


    val model: OrdinaryLeastSquaresModel[Int] = new OrdinaryLeastSquares[Int]().fit(drmX, drmy)
    //println(model.beta.toString)

    // Add in some extra tests here, but for now sufficient.
    model.fScore - 3.529130495  should be < epsilon
    model.r2 - 0.034424  should be < epsilon
    model.mse - 97.6567 should be < epsilon
    println(model.summary)

  }






}