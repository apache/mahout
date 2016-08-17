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
package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable

import org.apache.flink.api.common.functions._
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.shaded.com.google.common.collect.Lists
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.math.drm.logical.OpAtA
import org.apache.mahout.math.drm.{BlockifiedDrmTuple, _}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{Matrix, UpperTriangular, _}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection._

/**
 * Implementation of Flink A' * A
 *
 */
object FlinkOpAtA {

  final val PROPERTY_ATA_MAXINMEMNCOL = "mahout.math.AtA.maxInMemNCol"
  final val PROPERTY_ATA_MAXINMEMNCOL_DEFAULT = "200"

  def at_a[K](op: OpAtA[K], A: FlinkDrm[K]): FlinkDrm[Int] = {
    val maxInMemStr = System.getProperty(PROPERTY_ATA_MAXINMEMNCOL, PROPERTY_ATA_MAXINMEMNCOL_DEFAULT)
    val maxInMemNCol = maxInMemStr.toInt
    maxInMemNCol.ensuring(_ > 0, "Invalid A'A in-memory setting for optimizer")

    implicit val kTag = A.classTag

    if (op.ncol <= maxInMemNCol) {
      implicit val ctx = A.context
      val inCoreAtA = slim(op, A)
      val result = drmParallelize(inCoreAtA, numPartitions = 1)
      result
    } else {
      fat(op.asInstanceOf[OpAtA[K]], A.asInstanceOf[FlinkDrm[K]])
    }
  }

  def slim[K](op: OpAtA[K], A: FlinkDrm[K]): Matrix = {
    val ds = A.asRowWise.ds
    val ncol = op.ncol

    // Compute backing vector of tiny-upper-triangular accumulator across all the data.
    val res = ds.mapPartition(pIter => {

      val ut = new UpperTriangular(ncol)

      // Strategy is to add to an outer product of each row to the upper triangular accumulator.
      pIter.foreach({ case (k, v) =>

        // Use slightly various traversal strategies over dense vs. sparse source.
        if (v.isDense) {

          // Update upper-triangular pattern only (due to symmetry).
          // Note: Scala for-comprehensions are said to be fairly inefficient this way, but this is
          // such spectacular case they were designed for.. Yes I do observe some 20% difference
          // compared to while loops with no other payload, but the other paylxcoad is usually much
          // heavier than this overhead, so... I am keeping this as is for the time being.

          for (row <- 0 until v.length; col <- row until v.length)
            ut(row, col) = ut(row, col) + v(row) * v(col)

        } else {

          // Sparse source.
          v.nonZeroes().view

            // Outer iterator iterates over rows of outer product.
            .foreach(elrow => {

            // Inner loop for columns of outer product.
            v.nonZeroes().view

              // Filter out non-upper nonzero elements from the double loop.
              .filter(_.index >= elrow.index)

              // Incrementally update outer product value in the uppper triangular accumulator.
              .foreach(elcol => {

              val row = elrow.index
              val col = elcol.index
              ut(row, col) = ut(row, col) + elrow.get() * elcol.get()

            })
          })

        }
      })

      Iterator(dvec(ddata = ut.getData).asInstanceOf[Vector]: Vector)
    }).reduce(_ + _).collect()

    new DenseSymmetricMatrix(res.head)
  }

  def fat[K](op: OpAtA[K], A: FlinkDrm[K]): FlinkDrm[Int] = {
    val nrow = op.A.nrow
    val ncol = op.A.ncol
    val ds = A.asBlockified.ds

    val numberOfPartitions: DataSet[Int] = ds.map(new MapFunction[(Array[K], Matrix), Int] {
      def map(a: (Array[K], Matrix)): Int = 1
    }).reduce(new ReduceFunction[Int] {
      def reduce(a: Int, b: Int): Int = a + b
    })

    val subresults: DataSet[(Int, Matrix)] =
          ds.flatMap(new RichFlatMapFunction[(Array[K], Matrix), (Int, Matrix)] {

      var ranges: Array[Range] = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsX: java.util.List[Int] = runtime.getBroadcastVariable("numberOfPartitions")
        val parts = dsX.get(0)
        val numParts = estimatePartitions(nrow, ncol, parts)
        ranges = computeEvenSplits(ncol, numParts)
      }

      def flatMap(tuple: (Array[K], Matrix), out: Collector[(Int, Matrix)]): Unit = {
        val block = tuple._2

        ranges.zipWithIndex.foreach { case (range, idx) => 
          out.collect(idx -> block(::, range).t %*% block)
        }
      }

    }).withBroadcastSet(numberOfPartitions, "numberOfPartitions")

    val res = subresults.groupBy(0)
                        .reduceGroup(new RichGroupReduceFunction[(Int, Matrix), BlockifiedDrmTuple[Int]] {

      var ranges: Array[Range] = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsX: java.util.List[Int] = runtime.getBroadcastVariable("numberOfPartitions")
        val parts = dsX.get(0)
        val numParts = estimatePartitions(nrow, ncol, parts)
        ranges = computeEvenSplits(ncol, numParts)
      }

      def reduce(values: Iterable[(Int, Matrix)], out: Collector[BlockifiedDrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (blockKey, _) = it.head

        val block = it.map { _._2 }.reduce { (m1, m2) => m1 + m2 }
        val blockStart = ranges(blockKey).start
        val rowKeys = Array.tabulate(block.nrow)(blockStart + _)

        out.collect(rowKeys -> block)
      }
    }).withBroadcastSet(numberOfPartitions, "numberOfPartitions")

    new BlockifiedFlinkDrm(res, ncol)
  }

  def estimatePartitions(nrow: Long, ncol: Int, parts:Int): Int = {
    // per-partition element density 
    val epp = nrow.toDouble * ncol / parts

    // product partitions
    val prodParts = nrow * ncol / epp

    val nparts = math.round(prodParts).toInt max 1

    // Constrain nparts to maximum of anrow to prevent guaranteed empty partitions.
    if (nparts > nrow) { 
      nrow.toInt 
    } else { 
      nparts
    }
  }

  def computeEvenSplits(nrow: Long, numSplits: Int): Array[Range] = {
    require(numSplits <= nrow, "Requested amount of splits greater than number of data points.")
    require(nrow >= 1)
    require(numSplits >= 1)

    // Base split -- what is our base split size?
    val baseSplit = safeToNonNegInt(nrow / numSplits)

    // Slack -- how many splits will have to be incremented by 1 though?
    val slack = safeToNonNegInt(nrow % numSplits)

    // Compute ranges. We need to set ranges so that numSplits - slack splits have size of baseSplit;
    // and `slack` splits have size baseSplit + 1. Here is how we do it: First, we compute the range
    // offsets:
    val offsets = (0 to numSplits).map(i => i * (baseSplit + 1) - (0 max i - slack))
    // And then we connect the ranges using gaps between offsets:

    val ranges = offsets.sliding(2).map { offs => offs(0) until offs(1) }
    ranges.toArray
  }
}