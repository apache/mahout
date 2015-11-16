package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable

import scala.collection.JavaConverters._

import org.apache.flink.api.common.functions._
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.shaded.com.google.common.collect.Lists
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.BlockifiedDrmTuple
import org.apache.mahout.math.drm.logical.OpAtA
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._


/**
 * Inspired by Spark's implementation from 
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AtA.scala
 * 
 */
object FlinkOpAtA {

  final val PROPERTY_ATA_MAXINMEMNCOL = "mahout.math.AtA.maxInMemNCol"
  final val PROPERTY_ATA_MAXINMEMNCOL_DEFAULT = "200"

  def at_a(op: OpAtA[_], A: FlinkDrm[_]): FlinkDrm[Int] = {
    val maxInMemStr = System.getProperty(PROPERTY_ATA_MAXINMEMNCOL, PROPERTY_ATA_MAXINMEMNCOL_DEFAULT)
    val maxInMemNCol = maxInMemStr.toInt
    maxInMemNCol.ensuring(_ > 0, "Invalid A'A in-memory setting for optimizer")

    if (op.ncol <= maxInMemNCol) {
      implicit val ctx = A.context
      val inCoreAtA = slim(op, A)
      val result = drmParallelize(inCoreAtA, numPartitions = 1)
      result
    } else {
      fat(op.asInstanceOf[OpAtA[Any]], A.asInstanceOf[FlinkDrm[Any]])
    }
  }

  def slim(op: OpAtA[_], A: FlinkDrm[_]): Matrix = {
    val ds = A.asBlockified.ds.asInstanceOf[DataSet[(Array[Any], Matrix)]]

    val res = ds.map {
      // TODO: optimize it: use upper-triangle matrices like in Spark
      block => block._2.t %*% block._2
    }.reduce(_ + _).collect()

    res.head
  }

  def fat(op: OpAtA[Any], A: FlinkDrm[Any]): FlinkDrm[Int] = {
    val nrow = op.A.nrow
    val ncol = op.A.ncol
    val ds = A.asBlockified.ds

    val numberOfPartitions: DataSet[Int] = ds.map(new MapFunction[(Array[Any], Matrix), Int] {
      def map(a: (Array[Any], Matrix)): Int = 1
    }).reduce(new ReduceFunction[Int] {
      def reduce(a: Int, b: Int): Int = a + b
    })

    val subresults: DataSet[(Int, Matrix)] = 
          ds.flatMap(new RichFlatMapFunction[(Array[Any], Matrix), (Int, Matrix)] {

      var ranges: Array[Range] = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext()
        val dsX: java.util.List[Int] = runtime.getBroadcastVariable("numberOfPartitions")
        val parts = dsX.get(0)
        val numParts = estimatePartitions(nrow, ncol, parts)
        ranges = computeEvenSplits(ncol, numParts)
      }

      def flatMap(tuple: (Array[Any], Matrix), out: Collector[(Int, Matrix)]): Unit = {
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
        val runtime = this.getRuntimeContext()
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