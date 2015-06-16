package org.apache.mahout.flinkbindings.drm

import scala.reflect.ClassTag
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.flinkbindings._

import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.flinkbindings.FlinkDistributedContext
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.mahout.math.drm.CacheHint
import scala.util.Random
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.DenseMatrix
import org.apache.mahout.math.SparseMatrix
import org.apache.flink.api.java.io.LocalCollectionOutputFormat
import java.util.ArrayList

import scala.collection.JavaConverters._

class CheckpointedFlinkDrm[K: ClassTag](val ds: DrmDataSet[K],
  private var _nrow: Long = CheckpointedFlinkDrm.UNKNOWN,
  private var _ncol: Int = CheckpointedFlinkDrm.UNKNOWN,
  // private val _cacheStorageLevel: StorageLevel = StorageLevel.MEMORY_ONLY,
  override protected[mahout] val partitioningTag: Long = Random.nextLong(),
  private var _canHaveMissingRows: Boolean = false) extends CheckpointedDrm[K] {

  lazy val nrow = if (_nrow >= 0) _nrow else computeNRow
  lazy val ncol = if (_ncol >= 0) _ncol else computeNCol

  protected def computeNRow = ???
  protected def computeNCol = ??? /*{
  TODO: find out how to get one value
    val max = ds.map(new MapFunction[DrmTuple[K], Int] {
      def map(value: DrmTuple[K]): Int = value._2.length
    }).reduce(new ReduceFunction[Int] {
      def reduce(a1: Int, a2: Int) = Math.max(a1, a2)
    })
    
    max
  }*/
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
    val dataJavaList = new ArrayList[DrmTuple[K]]
    val outputFormat = new LocalCollectionOutputFormat[DrmTuple[K]](dataJavaList)
    ds.output(outputFormat)
    val data = dataJavaList.asScala
    ds.getExecutionEnvironment.execute("Checkpointed Flink Drm collect()")

    val isDense = data.forall(_._2.isDense)

    val m = if (isDense) {
      val cols = data.head._2.size()
      val rows = data.length
      new DenseMatrix(rows, cols)
    } else {
      val cols = ncol
      val rows = safeToNonNegInt(nrow)
      new SparseMatrix(rows, cols)
    }

    val intRowIndices = keyClassTag == implicitly[ClassTag[Int]]

    if (intRowIndices)
      data.foreach(t => m(t._1.asInstanceOf[Int], ::) := t._2)
    else {
      // assign all rows sequentially
      val d = data.zipWithIndex
      d.foreach(t => m(t._2, ::) := t._1._2)

      val rowBindings = d.map(t => (t._1._1.toString, t._2: java.lang.Integer)).toMap.asJava
      m.setRowLabelBindings(rowBindings)
    }

    m
  }

  def dfsWrite(path: String) = ???
  def newRowCardinality(n: Int): CheckpointedDrm[K] = ???

  override val context: DistributedContext = ds.getExecutionEnvironment

}

object CheckpointedFlinkDrm {
  val UNKNOWN = -1;
}