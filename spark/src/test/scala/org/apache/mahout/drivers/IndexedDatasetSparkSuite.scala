package org.apache.mahout.drivers

import org.apache.mahout.math.scalabindings.RLikeOps
import org.apache.mahout.math.{Vector, DenseVector, RandomAccessSparseVector}
import org.apache.mahout.math.indexeddataset.{BiDictionary, IndexedDataset}
import org.apache.mahout.sparkbindings.drm.CheckpointedDrmSpark
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import org.scalatest.{ConfigMap, FunSuite}
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import RLikeOps._
import org.scalatest.FunSuite
import org.apache.mahout.math.{RandomAccessSparseVector, Vector}
import RLikeOps._
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.math.scalabindings._

class IndexedDatasetSparkSuite extends FunSuite with DistributedSparkSuite {

  val items1 = Array(
    ("u1","iphone"),
    ("u1","ipad"),
    ("u2","nexus"),
    ("u2","galaxy"),
    ("u3","surface"),
    ("u4","iphone"),
    ("u4","galaxy"),
    ("u1","iphone"))

  val items2 = Array(
    ("u5","iphone"))

  val collectedDRM1 = Array(
    (0,new RandomAccessSparseVector(5) := (3 -> 1.0) :: (4 -> 1.0) :: Nil),
    (1,new RandomAccessSparseVector(5) := (2 -> 1.0) :: (1 -> 1.0) :: Nil),
    (2,new RandomAccessSparseVector(5) := (0 -> 1.0) :: Nil),
    (3,new RandomAccessSparseVector(5) := (2 -> 1.0) :: (4 -> 1.0) :: Nil))

  val collectedDRM2 = Array(
    (4,new RandomAccessSparseVector(5) := (0 -> 1.0) :: Nil))

  test("IndexedDatasetSpark constructors") {

    implicit val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc
    val itemsRDD1 = sc.parallelize(items1)
    val a1 = itemsRDD1.collect()
    val indexedDatasetSparkRDD1 = IndexedDatasetSpark(itemsRDD1)

    indexedDatasetSparkRDD1.rowIDs.size equals 4
    indexedDatasetSparkRDD1.columnIDs.size equals 5
    indexedDatasetSparkRDD1.matrix.nrow equals 4
    indexedDatasetSparkRDD1.matrix.ncol equals 5
    indexedDatasetSparkRDD1.matrix.rdd.collect() should contain theSameElementsAs collectedDRM1
    val anotherIDS = indexedDatasetSparkRDD1.create(indexedDatasetSparkRDD1.matrix,
      indexedDatasetSparkRDD1.rowIDs, indexedDatasetSparkRDD1.columnIDs)
    anotherIDS.rowIDs.size equals 4
    anotherIDS.columnIDs.size equals 5
    anotherIDS.matrix.nrow equals 4
    anotherIDS.matrix.ncol equals 5
    anotherIDS.matrix.rdd.collect() should contain theSameElementsAs collectedDRM1

    // now treat two IndexedDatasets as having only one user dictionary
    val itemsRDD2 = sc.parallelize(items2)
    val a2 = itemsRDD2.collect()
    val indexedDatasetSparkRDD2 = IndexedDatasetSpark(itemsRDD2, Some(indexedDatasetSparkRDD1.rowIDs))
    indexedDatasetSparkRDD2.rowIDs.size equals 5
    indexedDatasetSparkRDD2.columnIDs.size equals 5
    indexedDatasetSparkRDD2.matrix.nrow equals 5
    indexedDatasetSparkRDD2.matrix.ncol equals 5
    val d = indexedDatasetSparkRDD2.matrix.rdd.collect()
    indexedDatasetSparkRDD2.matrix.rdd.collect() equals collectedDRM2

  }

}
