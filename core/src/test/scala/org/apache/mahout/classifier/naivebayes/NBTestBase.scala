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
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.{FunSuite, Matchers}
import collection._
import JavaConversions._
import collection.JavaConversions

trait NBTestBase extends DistributedMahoutSuite with Matchers { this:FunSuite =>

  val epsilon = 1E-6

  test("Simple Standard NB Model") {

    // test from simulated sparse TF-IDF data
    val inCoreTFIDF = sparse(
      (0, 0.7) ::(1, 0.1) ::(2, 0.1) ::(3, 0.3) :: Nil,
      (0, 0.4) ::(1, 0.4) ::(2, 0.1) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.8) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.1) ::(2, 0.1) ::(3, 0.7) :: Nil
    )

    val TFIDFDrm = drm.drmParallelize(m = inCoreTFIDF, numPartitions = 2)

    val labelIndex = new java.util.HashMap[String,Integer]()
    labelIndex.put("Cat1", 3)
    labelIndex.put("Cat2", 2)
    labelIndex.put("Cat3", 1)
    labelIndex.put("Cat4", 0)

    // train a Standard NB Model
    val model = NaiveBayes.train(TFIDFDrm, labelIndex, false)

    // validate the model- will throw an exception if model is invalid
    model.validate()

    // check the labelWeights
    model.labelWeight(0) - 1.2 should be < epsilon
    model.labelWeight(1) - 1.0 should be < epsilon
    model.labelWeight(2) - 1.0 should be < epsilon
    model.labelWeight(3) - 1.0 should be < epsilon

    // check the Feature weights
    model.featureWeight(0) - 1.3 should be < epsilon
    model.featureWeight(1) - 0.6 should be < epsilon
    model.featureWeight(2) - 1.1 should be < epsilon
    model.featureWeight(3) - 1.2 should be < epsilon
  }

  test("NB Aggregator") {

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

    val (labelIndex, aggregatedTFIDFDrm) = NaiveBayes.extractLabelsAndAggregateObservations(TFIDFDrm)

    labelIndex.size should be (2)

    val cat1=labelIndex("Cat1")
    val cat2=labelIndex("Cat2")

    cat1 should be (0)
    cat2 should be (1)

    val aggregatedTFIDFInCore = aggregatedTFIDFDrm.collect
    aggregatedTFIDFInCore.numCols should be (4)
    aggregatedTFIDFInCore.numRows should be (2)

    aggregatedTFIDFInCore.get(cat1, 0) - 0.3 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 1) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 2) - 0.3 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 3) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 0) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 1) - 0.2 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 2) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 3) - 0.2 should be < epsilon

  }

  test("Model DFS Serialization") {

    // test from simulated sparse TF-IDF data
    val inCoreTFIDF = sparse(
      (0, 0.7) ::(1, 0.1) ::(2, 0.1) ::(3, 0.3) :: Nil,
      (0, 0.4) ::(1, 0.4) ::(2, 0.1) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.8) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.1) ::(2, 0.1) ::(3, 0.7) :: Nil
    )

    val labelIndex = new java.util.HashMap[String,Integer]()
    labelIndex.put("Cat1", 0)
    labelIndex.put("Cat2", 1)
    labelIndex.put("Cat3", 2)
    labelIndex.put("Cat4", 3)

    val TFIDFDrm = drm.drmParallelize(m = inCoreTFIDF, numPartitions = 2)

    // train a Standard NB Model- no label index here
    val model = NaiveBayes.train(TFIDFDrm, labelIndex, false)

    // validate the model- will throw an exception if model is invalid
    model.validate()

    // save the model
    model.dfsWrite(TmpDir)

    // reload a new model which should be equal to the original
    // this will automatically trigger a validate() call
    val materializedModel= NBModel.dfsRead(TmpDir)


    // check the labelWeights
    model.labelWeight(0) - materializedModel.labelWeight(0) should be < epsilon //1.2
    model.labelWeight(1) - materializedModel.labelWeight(1) should be < epsilon //1.0
    model.labelWeight(2) - materializedModel.labelWeight(2) should be < epsilon //1.0
    model.labelWeight(3) - materializedModel.labelWeight(3) should be < epsilon //1.0

    // check the Feature weights
    model.featureWeight(0) - materializedModel.featureWeight(0) should be < epsilon //1.3
    model.featureWeight(1) - materializedModel.featureWeight(1) should be < epsilon //0.6
    model.featureWeight(2) - materializedModel.featureWeight(2) should be < epsilon //1.1
    model.featureWeight(3) - materializedModel.featureWeight(3) should be < epsilon //1.2

    // check to se if the new model is complementary
    materializedModel.isComplementary should be (model.isComplementary)

    // check the label indexMaps
    for(elem <- model.labelIndex){
      model.labelIndex(elem._1) == materializedModel.labelIndex(elem._1) should be (true)
    }
  }

  test("train and test a model") {

    // test from simulated sparse TF-IDF data
    val inCoreTFIDF = sparse(
      (0, 0.7) ::(1, 0.1) ::(2, 0.1) ::(3, 0.3) :: Nil,
      (0, 0.4) ::(1, 0.4) ::(2, 0.1) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.8) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.1) ::(2, 0.1) ::(3, 0.7) :: Nil
    )

    val labelIndex = new java.util.HashMap[String,Integer]()
    labelIndex.put("/Cat1/", 0)
    labelIndex.put("/Cat2/", 1)
    labelIndex.put("/Cat3/", 2)
    labelIndex.put("/Cat4/", 3)

    val TFIDFDrm = drm.drmParallelize(m = inCoreTFIDF, numPartitions = 2)

    // train a Standard NB Model- no label index here
    val model = NaiveBayes.train(TFIDFDrm, labelIndex, false)

    // validate the model- will throw an exception if model is invalid
    model.validate()

    // save the model
    model.dfsWrite(TmpDir)

    // reload a new model which should be equal to the original
    // this will automatically trigger a validate() call
    val materializedModel= NBModel.dfsRead(TmpDir)


    // check to se if the new model is complementary
    materializedModel.isComplementary should be (model.isComplementary)

    // check the label indexMaps
    for(elem <- model.labelIndex){
      model.labelIndex(elem._1) == materializedModel.labelIndex(elem._1) should be (true)
    }


    //self test on the original set
    val inCoreTFIDFWithLabels = inCoreTFIDF.clone()
    inCoreTFIDFWithLabels.setRowLabelBindings(labelIndex)
    val TFIDFDrmWithLabels = drm.drmParallelizeWithRowLabels(m = inCoreTFIDFWithLabels, numPartitions = 2)

    NaiveBayes.test(materializedModel,TFIDFDrmWithLabels , false)

  }

  test("train and test a model with the confusion matrix") {

    val rowBindings = new java.util.HashMap[String,Integer]()
    rowBindings.put("/Cat1/doc_a/", 0)
    rowBindings.put("/Cat2/doc_b/", 1)
    rowBindings.put("/Cat1/doc_c/", 2)
    rowBindings.put("/Cat2/doc_d/", 3)
    rowBindings.put("/Cat1/doc_e/", 4)
    rowBindings.put("/Cat2/doc_f/", 5)
    rowBindings.put("/Cat1/doc_g/", 6)
    rowBindings.put("/Cat2/doc_h/", 7)
    rowBindings.put("/Cat1/doc_i/", 8)
    rowBindings.put("/Cat2/doc_j/", 9)

    val seed = 1

    val matrixSetup = Matrices.uniformView(10, 50 , seed)

    println("TFIDF matrix")
    println(matrixSetup)

    matrixSetup.setRowLabelBindings(rowBindings)

    val TFIDFDrm = drm.drmParallelizeWithRowLabels(matrixSetup)

  //  println("Parallelized and Collected")
  //  println(TFIDFDrm.collect)

    val (labelIndex, aggregatedTFIDFDrm) = NaiveBayes.extractLabelsAndAggregateObservations(TFIDFDrm)

    println("Aggregated by key")
    println(aggregatedTFIDFDrm.collect)
    println(labelIndex)


    // train a Standard NB Model- no label index here
    val model = NaiveBayes.train(aggregatedTFIDFDrm, labelIndex, false)

    // validate the model- will throw an exception if model is invalid
    model.validate()

    // save the model
    model.dfsWrite(TmpDir)

    // reload a new model which should be equal to the original
    // this will automatically trigger a validate() call
    val materializedModel= NBModel.dfsRead(TmpDir)

    // check to se if the new model is complementary
    materializedModel.isComplementary should be (model.isComplementary)

    // check the label indexMaps
    for(elem <- model.labelIndex){
      model.labelIndex(elem._1) == materializedModel.labelIndex(elem._1) should be (true)
    }

 //   val testTFIDFDrm = drm.drmParallelizeWithRowLabels(m = matrixSetup, numPartitions = 2)

    // self test on this model
    val result = NaiveBayes.test(materializedModel, TFIDFDrm , false)

    println(result)

    result.getConfusionMatrix.getMatrix.getQuick(0, 0) should be(5)
    result.getConfusionMatrix.getMatrix.getQuick(0, 1) should be(0)
    result.getConfusionMatrix.getMatrix.getQuick(1, 0) should be(0)
    result.getConfusionMatrix.getMatrix.getQuick(1, 1) should be(5)

  }

}
