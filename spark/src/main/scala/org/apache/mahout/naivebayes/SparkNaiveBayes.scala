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

package org.apache.mahout.naivebayes

import org.apache.mahout.math._

import org.apache.spark.rdd.RDD
import org.apache.mahout.sparkbindings.drm.{CheckpointedDrmSpark, DrmRddInput}

import scalabindings._
import scalabindings.RLikeOps._
import drm.RLikeDrmOps._
import drm._
import scala.reflect.ClassTag
import scala.language.asInstanceOf
import collection._
import JavaConversions._
import org.apache.spark.SparkContext._

import org.apache.mahout.classifier.naivebayes._
import org.apache.mahout.sparkbindings._

/**
 * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
 * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
 */
object SparkNaiveBayes extends NaiveBayes{

  /**
   * Extract label Keys from raw TF or TF-IDF Matrix generated by seqdirectory/seq2sparse
   * and aggregate TF or TF-IDF values by their label
   *
   * Optimized for spark
   *
   * @param stringKeyedObservations DrmLike matrix; Output from seq2sparse
   *   in form K= eg./Category/document_title
   *           V= TF or TF-IDF values per term
   * @param cParser a String => String function used to extract categories from
   *   Keys of the stringKeyedObservations DRM. The default
   *   CategoryParser will extract "Category" from: '/Category/document_id'
   * @return  (labelIndexMap,aggregatedByLabelObservationDrm)
   *   labelIndexMap is a HashMap  K= label row index
   *                               V= label
   *   aggregatedByLabelObservationDrm is a DrmLike[Int] of aggregated
   *   TF or TF-IDF counts per label
   */
  override def extractLabelsAndAggregateObservations[K: ClassTag]
    (stringKeyedObservations: DrmLike[K], cParser: CategoryParser = seq2SparseCategoryParser):
    (mutable.HashMap[String,Double], DrmLike[Int]) = {

    implicit val distributedContext = stringKeyedObservations.context

    val stringKeyedRdd = stringKeyedObservations
                           .checkpoint()
                           .asInstanceOf[CheckpointedDrmSpark[String]]
                           .rdd

    // is it necessary to sort this?
    // how expensive is it for spark to sort (relatively small) tuples?
    val aggregatedRdd= stringKeyedRdd
                         .map(x => (cParser(x._1), x._2))
                         .reduceByKey(_ + _)
                         .sortByKey(true)

    stringKeyedObservations.uncache()

    var categoryIndex = 0.0d
    val categoryMap = new mutable.HashMap[String, Double]

    // has to be an better way of creating this map
    val categoryArray=aggregatedRdd.keys.takeOrdered(aggregatedRdd.count.toInt)
    for(i <- 0 until categoryArray.size){
      categoryMap.put(categoryArray(i), categoryIndex)
      categoryIndex = categoryIndex+ 1.0
    }

    val intKeyedRdd = aggregatedRdd.map(x => (categoryMap(x._1).toInt, x._2))

    val aggregetedObservationByLabelDrm = drmWrap(intKeyedRdd)

    (categoryMap, aggregetedObservationByLabelDrm)
  }


}

