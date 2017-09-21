/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */

package org.apache.mahout.sparkbindings.algorithms.regression

import org.apache.mahout.sparkbindings.test._
import org.apache.mahout.test.MahoutSuite

import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession, SQLContext}

import org.scalatest.Matchers._
import org.scalatest.FunSuite

import scala.collection.JavaConversions._

case class MiniPanda(happy: Double, fuzzy: Double, old: Double)

class OrdinaryLeastSquaresPipelineSuite extends FunSuite with MahoutSuite with DistributedSparkSuite  {

  val miniPandasList = List(
    MiniPanda(1.0, 1.0, 1.0),
    MiniPanda(1.0, 1.0, 0.0),
    MiniPanda(1.0, 1.0, 0.0),
    MiniPanda(0.0, 0.0, 1.0),
    MiniPanda(0.0, 0.0, 0.0))

  test("basic train test") {
    val session = SparkSession.builder().getOrCreate()
    import session.implicits._
    val ds: Dataset[MiniPanda] = session.createDataset(miniPandasList)
    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("fuzzy", "old"))
    assembler.setOutputCol("magical_features")
    val ols = new OrdinaryLeastSquaresEstimator()
    ols.setFeaturesCol("magical_features")
    ols.setLabelCol("happy")
    val pipeline = new Pipeline().setStages(Array(assembler, ols))
    val model = pipeline.fit(ds)
    val test = ds.select("fuzzy", "old")
    val predicted = model.transform(test)
    assert(predicted.count() === miniPandasList.size)
  }
}
