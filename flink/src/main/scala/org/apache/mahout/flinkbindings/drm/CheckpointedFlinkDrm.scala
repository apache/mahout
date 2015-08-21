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
package org.apache.mahout.flinkbindings.drm

import scala.collection.JavaConverters._
import scala.util.Random
import scala.reflect.ClassTag
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.common.functions.ReduceFunction
import org.apache.flink.api.java.hadoop.mapred.HadoopOutputFormat
import org.apache.flink.api.java.tuple.Tuple2
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.io.LongWritable
import org.apache.hadoop.io.Text
import org.apache.hadoop.io.Writable
import org.apache.hadoop.mapred.FileOutputFormat
import org.apache.hadoop.mapred.JobConf
import org.apache.hadoop.mapred.SequenceFileOutputFormat
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.DrmDataSet
import org.apache.mahout.math.DenseMatrix
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.SparseMatrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.VectorWritable
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.CacheHint
import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.drm.DrmTuple
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._

class CheckpointedFlinkDrm[K: ClassTag](val ds: DrmDataSet[K],
      private var _nrow: Long = CheckpointedFlinkDrm.UNKNOWN,
      private var _ncol: Int = CheckpointedFlinkDrm.UNKNOWN,
      override protected[mahout] val partitioningTag: Long = Random.nextLong(),
      private var _canHaveMissingRows: Boolean = false
  ) extends CheckpointedDrm[K] {

  lazy val nrow: Long = if (_nrow >= 0) _nrow else dim._1
  lazy val ncol: Int = if (_ncol >= 0) _ncol else dim._2

  private lazy val dim: (Long, Int) = {
    // combine computation of ncol and nrow in one pass

    val res = ds.map(new MapFunction[DrmTuple[K], (Long, Int)] {
      def map(value: DrmTuple[K]): (Long, Int) = {
        (1L, value._2.length)
      }
    }).reduce(new ReduceFunction[(Long, Int)] {
      def reduce(t1: (Long, Int), t2: (Long, Int)) = {
        val ((rowCnt1, colNum1), (rowCnt2, colNum2)) = (t1, t2)
        (rowCnt1 + rowCnt2, Math.max(colNum1, colNum2))
      }
    })

    val list = res.collect().asScala.toList
    list.head
  }

  def keyClassTag: ClassTag[K] = implicitly[ClassTag[K]]

  def cache() = {
    // TODO
    this
  }

  def uncache = ???

  // Members declared in org.apache.mahout.math.drm.DrmLike   

  protected[mahout] def canHaveMissingRows: Boolean = _canHaveMissingRows

  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = this

  def collect: Matrix = {
    val data = ds.collect().asScala.toList
    val isDense = data.forall(_._2.isDense)

    val cols = ncol
    val rows = safeToNonNegInt(nrow)

    val m = if (isDense) {
      new DenseMatrix(rows, cols)
    } else {
      new SparseMatrix(rows, cols)
    }

    val intRowIndices = keyClassTag == implicitly[ClassTag[Int]]

    if (intRowIndices) {
      data.foreach { case (t, vec) =>
        val idx = t.asInstanceOf[Int]
        m(idx, ::) := vec
      }

      println(m.ncol, m.nrow)
    } else {
      // assign all rows sequentially
      val d = data.zipWithIndex
      d.foreach {
        case ((_, vec), idx) => m(idx, ::) := vec
      }

      val rowBindings = d.map {
        case ((t, _), idx) => (t.toString, idx: java.lang.Integer) 
      }.toMap.asJava

      m.setRowLabelBindings(rowBindings)
    }

    m
  }

  def dfsWrite(path: String): Unit = {
    val env = ds.getExecutionEnvironment

    val keyTag = implicitly[ClassTag[K]]
    val convertKey = keyToWritableFunc(keyTag)

    val writableDataset = ds.map(new MapFunction[(K, Vector), Tuple2[Writable, VectorWritable]] {
      def map(tuple: (K, Vector)): Tuple2[Writable, VectorWritable] = tuple match {
        case (idx, vec) => new Tuple2(convertKey(idx), new VectorWritable(vec))
      }
    })

    val job = new JobConf
    val sequenceFormat = new SequenceFileOutputFormat[Writable, VectorWritable]
    FileOutputFormat.setOutputPath(job, new org.apache.hadoop.fs.Path(path))

    val hadoopOutput  = new HadoopOutputFormat(sequenceFormat, job)
    writableDataset.output(hadoopOutput)

    env.execute(s"dfsWrite($path)")
  }

  private def keyToWritableFunc[K: ClassTag](keyTag: ClassTag[K]): (K) => Writable = {
    if (keyTag.runtimeClass == classOf[Int]) { 
      (x: K) => new IntWritable(x.asInstanceOf[Int])
    } else if (keyTag.runtimeClass == classOf[String]) {
      (x: K) => new Text(x.asInstanceOf[String]) 
    } else if (keyTag.runtimeClass == classOf[Long]) {
      (x: K) => new LongWritable(x.asInstanceOf[Long]) 
    } else if (classOf[Writable].isAssignableFrom(keyTag.runtimeClass)) { 
      (x: K) => x.asInstanceOf[Writable] 
    } else { 
      throw new IllegalArgumentException("Do not know how to convert class tag %s to Writable.".format(keyTag))
    }
  }

  def newRowCardinality(n: Int): CheckpointedDrm[K] = ???

  override val context: DistributedContext = ds.getExecutionEnvironment

}

object CheckpointedFlinkDrm {
  val UNKNOWN = -1

}