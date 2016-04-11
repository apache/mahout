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

import org.apache.flink.api.common.functions.{MapFunction, ReduceFunction}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.io.{TypeSerializerInputFormat, TypeSerializerOutputFormat}
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.core.fs.Path
import org.apache.flink.api.scala.hadoop.mapred.HadoopOutputFormat
import org.apache.flink.configuration.GlobalConfiguration
import org.apache.hadoop.io.{IntWritable, LongWritable, Text, Writable}
import org.apache.hadoop.mapred.{FileOutputFormat, JobConf, SequenceFileOutputFormat}
import org.apache.mahout.flinkbindings.io.Hadoop2HDFSUtil
import org.apache.mahout.flinkbindings.{DrmDataSet, _}
import org.apache.mahout.math._
import org.apache.mahout.math.drm.CacheHint._
import org.apache.mahout.math.drm.{CacheHint, CheckpointedDrm, DistributedContext, DrmTuple, _}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.collection.JavaConverters._
import scala.reflect.{ClassTag, classTag}
import scala.util.Random

class CheckpointedFlinkDrm[K: ClassTag:TypeInformation](val ds: DrmDataSet[K],
      private var _nrow: Long = CheckpointedFlinkDrm.UNKNOWN,
      private var _ncol: Int = CheckpointedFlinkDrm.UNKNOWN,
      override val cacheHint: CacheHint = CacheHint.NONE,
      override protected[mahout] val partitioningTag: Long = Random.nextLong(),
      private var _canHaveMissingRows: Boolean = false
  ) extends CheckpointedDrm[K] {

  lazy val nrow: Long = if (_nrow >= 0) _nrow else dim._1
  lazy val ncol: Int = if (_ncol >= 0) _ncol else dim._2

  // persistance values
  var cacheFileName: String = "undefinedCacheName"
  var isCached: Boolean = false
  var parallelismDeg: Int = -1
  var persistanceRootDir: String = _

  // need to make sure that this is actually getting the correct properties for {{taskmanager.tmp.dirs}}
  val mahoutHome = getMahoutHome()

  // this is extra I/O for each cache call.  this needs to be moved somewhere where it is called
  // only once.  Possibly FlinkDistributedEngine.
  GlobalConfiguration.loadConfiguration(mahoutHome + "/conf/flink-config.yaml")

  val conf = GlobalConfiguration.getConfiguration

  if (!(conf == null )) {
     persistanceRootDir = conf.getString("taskmanager.tmp.dirs", "/tmp")
  } else {
     persistanceRootDir = "/tmp"
  }


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

    val list = res.collect()
    list.head
  }


  override val keyClassTag: ClassTag[K] = classTag[K]

  /** Note as of Flink 1.0.0, no direct flink caching exists so we save
    * the dataset to the filesystem and read it back when cache is called */
  def cache() = {
    if (!isCached) {
      cacheFileName = persistanceRootDir + "/" + System.nanoTime().toString
      parallelismDeg = ds.getParallelism
      isCached = true
      persist(ds, cacheFileName)
    }
    val _ds = readPersistedDataSet(cacheFileName, ds)

    /** Leave the parallelism degree to be set the operators
      * TODO: find out a way to set the parallelism degree based on the
      * final drm after computation is actually triggered
      *
      *  // We may want to look more closely at this:
      *  // since we've cached a drm, triggering a computation
      *  // it may not make sense to keep the same parallelism degree
      *  if (!(parallelismDeg == _ds.getParallelism)) {
      *    _ds.setParallelism(parallelismDeg).rebalance()
      *  }
      *
      */

    datasetWrap(_ds)
  }

  def uncache(): this.type = {
    if (isCached) {
      Hadoop2HDFSUtil.delete(cacheFileName)
      isCached = false
    }
    this
  }

  /** Writes a [[DataSet]] to the specified path and returns it as a DataSource for subsequent
    * operations.
    *
    * @param dataset [[DataSet]] to write to disk
    * @param path File path to write dataset to
    * @tparam T Type of the [[DataSet]] elements
    */
  def persist[T: ClassTag: TypeInformation](dataset: DataSet[T], path: String): Unit = {
    val env = dataset.getExecutionEnvironment
    val outputFormat = new TypeSerializerOutputFormat[T]
    val filePath = new Path(path)

    outputFormat.setOutputFilePath(filePath)
    outputFormat.setWriteMode(WriteMode.OVERWRITE)

    dataset.output(outputFormat)
    env.execute("FlinkTools persist")
  }

  /** Read a [[DataSet]] from specified path and returns it as a DataSource for subsequent
    * operations.
    *
    * @param path File path to read dataset from
    * @param ds persisted ds to retrieve type information and environment forom
    * @tparam T key Type of the [[DataSet]] elements
    * @return [[DataSet]] the persisted dataset
    */
  def readPersistedDataSet[T: ClassTag : TypeInformation]
       (path: String, ds: DataSet[T]): DataSet[T] = {

    val env = ds.getExecutionEnvironment
    val inputFormat = new TypeSerializerInputFormat[T](ds.getType())
    val filePath = new Path(path)
    inputFormat.setFilePath(filePath)

    env.createInput(inputFormat)
  }


  // Members declared in org.apache.mahout.math.drm.DrmLike

  protected[mahout] def canHaveMissingRows: Boolean = _canHaveMissingRows

  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = {
    this
  }

  def collect: Matrix = {
    val data = ds.collect()
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

    val job = new JobConf
    FileOutputFormat.setOutputPath(job, new org.apache.hadoop.fs.Path(path))

    // explicitly define all Writable Subclasses for ds.map() keys
    // as well as the SequenceFileOutputFormat paramaters
    if (keyTag.runtimeClass == classOf[Int]) {
      // explicitly map into Int keys
      implicit val typeInformation = createTypeInformation[(IntWritable,VectorWritable)]
      val writableDataset = ds.map(new MapFunction[DrmTuple[K], (IntWritable, VectorWritable)] {
        def map(tuple: DrmTuple[K]): (IntWritable, VectorWritable) =
          (new IntWritable(tuple._1.asInstanceOf[Int]), new VectorWritable(tuple._2))
      })

      // setup sink for IntWritable
      job.setOutputKeyClass(classOf[IntWritable])
      job.setOutputValueClass(classOf[VectorWritable])
      val sequenceFormat = new SequenceFileOutputFormat[IntWritable, VectorWritable]
      val hadoopOutput  = new HadoopOutputFormat(sequenceFormat, job)
      writableDataset.output(hadoopOutput)

     } else if (keyTag.runtimeClass == classOf[String]) {
      // explicitly map into Text keys
      val writableDataset = ds.map(new MapFunction[DrmTuple[K], (Text, VectorWritable)] {
        def map(tuple: DrmTuple[K]): (Text, VectorWritable) =
          (new Text(tuple._1.asInstanceOf[String]), new VectorWritable(tuple._2))
      })

      // setup sink for Text
      job.setOutputKeyClass(classOf[Text])
      job.setOutputValueClass(classOf[VectorWritable])
      val sequenceFormat = new SequenceFileOutputFormat[Text, VectorWritable]
      val hadoopOutput  = new HadoopOutputFormat(sequenceFormat, job)
      writableDataset.output(hadoopOutput)

    } else if (keyTag.runtimeClass == classOf[Long]) {
      // explicitly map into Long keys
      val writableDataset = ds.map(new MapFunction[DrmTuple[K], (LongWritable, VectorWritable)] {
        def map(tuple: DrmTuple[K]): (LongWritable, VectorWritable) =
          (new LongWritable(tuple._1.asInstanceOf[Long]), new VectorWritable(tuple._2))
      })

      // setup sink for LongWritable
      job.setOutputKeyClass(classOf[LongWritable])
      job.setOutputValueClass(classOf[VectorWritable])
      val sequenceFormat = new SequenceFileOutputFormat[LongWritable, VectorWritable]
      val hadoopOutput  = new HadoopOutputFormat(sequenceFormat, job)
      writableDataset.output(hadoopOutput)

    } else throw new IllegalArgumentException("Do not know how to convert class tag %s to Writable.".format(keyTag))

    env.execute(s"dfsWrite($path)")
  }

  private def keyToWritableFunc[K: ClassTag](keyTag: ClassTag[K]): (K) => Writable = {
    if (keyTag.runtimeClass == classOf[Int]) { 
      (x: K) => new IntWritable(x.asInstanceOf[Int])
    } else if (keyTag.runtimeClass == classOf[String]) {
      (x: K) => new Text(x.asInstanceOf[String]) 
    } else if (keyTag.runtimeClass == classOf[Long]) {
      (x: K) => new LongWritable(x.asInstanceOf[Long])
    } else {
      throw new IllegalArgumentException("Do not know how to convert class tag %s to Writable.".format(keyTag))
    }
  }

  def newRowCardinality(n: Int): CheckpointedDrm[K] = {
    assert(n > -1)
    assert(n >= nrow)
    new CheckpointedFlinkDrm(ds = ds, _nrow = n, _ncol = _ncol, cacheHint = cacheHint,
      partitioningTag = partitioningTag, _canHaveMissingRows = _canHaveMissingRows)
  }

  override val context: DistributedContext = ds.getExecutionEnvironment

}

object CheckpointedFlinkDrm {
  val UNKNOWN = -1

}