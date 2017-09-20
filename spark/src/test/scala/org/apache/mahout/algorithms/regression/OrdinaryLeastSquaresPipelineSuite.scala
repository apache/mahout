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

import com.holdenkarau.spark.testing._

import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext}

import org.scalatest.Matchers._
import org.scalatest.FunSuite

case class MiniPanda(happy: Double, fuzzy: Double, old: Double)

class OrdinaryLeastSquaresPipelineSuite extends FunSuite with DataFrameSuiteBase {

  override protected implicit def reuseContextIfPossible: Boolean = false

  override protected implicit def enableHiveSupport: Boolean = false

  override def conf = {
    new SparkConf().
      setMaster(System.getProperties.getOrElse("test.spark.master", "local[4]")).
      setAppName("test").
      set("spark.ui.enabled", "false").
      set("spark.app.id", appID).
      set("spark.kryo.referenceTracking", "false").
      set("spark.kryo.registrator", "org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator").
      set("spark.kryoserializer.buffer", "32").
      set("spark.kryoserializer.buffer.max", "600m").
      set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  }

  val miniPandasList = List(
    MiniPanda(1.0, 1.0, 1.0),
    MiniPanda(1.0, 1.0, 0.0),
    MiniPanda(1.0, 1.0, 0.0),
    MiniPanda(0.0, 0.0, 1.0),
    MiniPanda(0.0, 0.0, 0.0))

  test("basic train test") {
    val session = spark
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
