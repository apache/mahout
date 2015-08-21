package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.reflect.ClassTag
import org.apache.mahout.math.drm._
import org.apache.flink.api.common.functions.CoGroupFunction
import org.apache.flink.api.java.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math._
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.scalabindings.RLikeOps._
import com.google.common.collect.Lists
import org.apache.flink.shaded.com.google.common.collect.Lists
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.SequentialAccessSparseVector
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.DrmTuple
import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.common.functions.ReduceFunction


/**
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
      fat(op, A)
    }
  }

  def slim(op: OpAtA[_], A: FlinkDrm[_]): Matrix = {
    val ds = A.blockify.ds.asInstanceOf[DataSet[(Array[Any], Matrix)]]

    val res = ds.map(new MapFunction[(Array[Any], Matrix), Matrix] {
      // TODO: optimize it: use upper-triangle matrices like in Spark
      def map(block: (Array[Any], Matrix)): Matrix =  block match {
        case (idx, m) => m.t %*% m
      }
    }).reduce(new ReduceFunction[Matrix] {
      def reduce(m1: Matrix, m2: Matrix) = m1 + m2
    }).collect()

    res.asScala.head
  }

  def fat(op: OpAtA[_], A: FlinkDrm[_]): FlinkDrm[Int] = {
    throw new NotImplementedError("fat matrices are not yet supported")
  }
}
