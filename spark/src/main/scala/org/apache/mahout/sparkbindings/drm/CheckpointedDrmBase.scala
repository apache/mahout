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

import org.apache.mahout.math.{SparseMatrix, DenseMatrix, Matrix, Vector}
import math._
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import scala.collection.JavaConversions._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext._
import reflect._
import scala.util.Random
import org.apache.hadoop.io.{LongWritable, Text, IntWritable, Writable}

class CheckpointedDrmBase[K: ClassTag](
    val rdd: DrmRdd[K],
    private var _nrow: Long = -1L,
    private var _ncol: Int = -1,
    private val _cacheStorageLevel: StorageLevel = StorageLevel.MEMORY_ONLY,
    private[sparkbindings] val partitioningTag: Long = Random.nextLong()

    ) extends CheckpointedDrm[K] {


  lazy val nrow = if (_nrow >= 0) _nrow else computeNRow
  lazy val ncol = if (_ncol >= 0) _ncol else computeNCol

  private var cached: Boolean = false


  /**
   * Action operator -- does not necessary means Spark action; but does mean running BLAS optimizer
   * and writing down Spark graph lineage since last checkpointed DRM.
   */
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] =
  // We are already checkpointed in a sense that we already have Spark lineage. So just return self.
    this

  def cache() = {
    if (!cached) {
      rdd.persist(_cacheStorageLevel)
      cached = true
    }
    this
  }


  /**
   * if matrix was previously persisted into cache,
   * delete cached representation
   */
  def uncache() = {
    if (cached) {
      rdd.unpersist(blocking = false)
      cached = false
    }
    this
  }

  def mapRows(mapfun: (K, Vector) => Vector): CheckpointedDrmBase[K] =
    new CheckpointedDrmBase[K](rdd.map(t => (t._1, mapfun(t._1, t._2))))


  /**
   * Collecting DRM to fron-end in-core Matrix.
   *
   * If key in DRM is Int, then matrix is collected using key as row index.
   * Otherwise, order of rows in result is undefined but key.toString is applied
   * as rowLabelBindings of the in-core matrix .
   *
   * Note that this pre-allocates target matrix and then assigns collected RDD to it
   * thus this likely would require about 2 times the RDD memory
   * @return
   */
  def collect: Matrix = {

    val intRowIndices = implicitly[ClassTag[K]] == implicitly[ClassTag[Int]]

    val cols = rdd.map(_._2.length).fold(0)(max(_, _))
    val rows = if (intRowIndices) rdd.map(_._1.asInstanceOf[Int]).fold(-1)(max(_, _)) + 1 else rdd.count().toInt

    // since currently spark #collect() requires Serializeable support,
    // we serialize DRM vectors into byte arrays on backend and restore Vector
    // instances on the front end:
    val data = rdd.map(t => (t._1, t._2)).collect()


    val m = if (data.forall(_._2.isDense))
      new DenseMatrix(rows, cols)

    else
      new SparseMatrix(rows, cols)

    if (intRowIndices)
      data.foreach(t => m(t._1.asInstanceOf[Int], ::) := t._2)
    else {

      // assign all rows sequentially
      val d = data.zipWithIndex
      d.foreach(t => m(t._2, ::) := t._1._2)

      // row bindings
      val rowBindings = d.map(t => (t._1._1.toString, t._2: java.lang.Integer)).toMap

      m.setRowLabelBindings(rowBindings)
    }

    m
  }


  /**
   * Dump matrix as computed Mahout's DRM into specified (HD)FS path
   * @param path
   */
  def writeDRM(path: String) = {
    val ktag = implicitly[ClassTag[K]]

    implicit val k2wFunc: (K) => Writable =
      if (ktag.runtimeClass == classOf[Int]) (x: K) => new IntWritable(x.asInstanceOf[Int])
      else if (ktag.runtimeClass == classOf[String]) (x: K) => new Text(x.asInstanceOf[String])
      else if (ktag.runtimeClass == classOf[Long]) (x: K) => new LongWritable(x.asInstanceOf[Long])
      else if (classOf[Writable].isAssignableFrom(ktag.runtimeClass)) (x: K) => x.asInstanceOf[Writable]
      else throw new IllegalArgumentException("Do not know how to convert class tag %s to Writable.".format(ktag))
//    implicit def any2w(k: Any): Writable = k2wFunc(k)
    rdd.saveAsSequenceFile(path)
  }

  protected def computeNRow = {

    val intRowIndex = classTag[K] == classTag[Int]

    if (intRowIndex)
      cache().rdd.map(_._1.asInstanceOf[Int]).fold(-1)(max(_, _)) + 1L
    else
      cache().rdd.count()
  }

  protected def computeNCol =
    cache().rdd.map(_._2.length).fold(-1)(max(_, _))

  protected def computeNNonZero =
    cache().rdd.map(_._2.getNumNonZeroElements.toLong).sum().toLong

}
