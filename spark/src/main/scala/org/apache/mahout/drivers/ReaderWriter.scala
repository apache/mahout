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

package org.apache.mahout.drivers

import org.apache.spark.SparkContext._
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.spark.SparkContext
import com.google.common.collect.{BiMap, HashBiMap}
import scala.collection.JavaConversions._
import org.apache.mahout.math.drm.{CheckpointedDrm, DrmLike}
import org.apache.mahout.sparkbindings._


/** Reader trait is abstract in the sense that the reader function must be defined by an extending trait, which also defines the type to be read.
  * @tparam T type of object read, usually supplied by an extending trait.
  * @todo the reader need not create both dictionaries but does at present. There are cases where one or the other dictionary is never used so saving the memory for a very large dictionary may be worth the optimization to specify which dictionaries are created.
  */
trait Reader[T]{
  val mc: SparkContext
  val readSchema: Schema
  protected def reader(mc: SparkContext, readSchema: Schema, source: String): T
  def readFrom(source: String): T = reader(mc, readSchema, source)
}

/** Writer trait is abstract in the sense that the writer method must be supplied by an extending trait, which also defines the type to be written.
  * @tparam T
  */
trait Writer[T]{
  val mc: SparkContext
  val writeSchema: Schema
  protected def writer(mc: SparkContext, writeSchema: Schema, dest: String, collection: T): Unit
  def writeTo(collection: T, dest: String) = writer(mc, writeSchema, dest, collection)
}

/** Extends Reader trait to supply the [[org.apache.mahout.drivers.IndexedDataset]] as the type read and a reader function for reading text delimited files as described in the [[org.apache.mahout.drivers.Schema]]
  */
trait TDIndexedDatasetReader extends Reader[IndexedDataset]{
  /** Read in text delimited tuples from all URIs in this comma delimited source String. 
    * 
    * @param mc context for the Spark job
    * @param readSchema describes the delimiters and positions of values in the text delimited file.
    * @param source comma delimited URIs of text files to be read into the [[org.apache.mahout.drivers.IndexedDataset]]
    * @return
    */
  protected def reader(mc: SparkContext, readSchema: Schema, source: String): IndexedDataset = {
    try {
      val delimiter = readSchema("delim").asInstanceOf[String]
      val rowIDPosition = readSchema("rowIDPosition").asInstanceOf[Int]
      val columnIDPosition = readSchema("columnIDPosition").asInstanceOf[Int]
      val filterPosition = readSchema("filterPosition").asInstanceOf[Int]
      val filterBy = readSchema("filter").asInstanceOf[String]
      //instance vars must be put into locally scoped vals when used in closures that are
      //executed but Spark

      assert(!source.isEmpty, {
        println(this.getClass.toString + ": has no files to read")
        throw new IllegalArgumentException
      })

      var columns = mc.textFile(source).map({ line => line.split(delimiter)})

      columns = columns.filter({ tokens => tokens(filterPosition) == filterBy})

      val interactions = columns.map({ tokens => tokens(rowIDPosition) -> tokens(columnIDPosition)})

      interactions.cache()

      val rowIDs = interactions.map({ case (rowID, _) => rowID}).distinct().collect()
      val columnIDs = interactions.map({ case (_, columnID) => columnID}).distinct().collect()

      val numRows = rowIDs.size
      val numColumns = columnIDs.size

      val rowIDDictionary = asOrderedDictionary(rowIDs)
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = asOrderedDictionary(columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val indexedInteractions =
        interactions.map({ case (rowID, columnID) =>
          val rowIndex = rowIDDictionary_bcast.value.get(rowID).get
          val columnIndex = columnIDDictionary_bcast.value.get(columnID).get

          rowIndex -> columnIndex
        }).groupByKey().map({ case (rowIndex, columnIndexes) =>
          val row = new RandomAccessSparseVector(numColumns)
          for (columnIndex <- columnIndexes) {
            row.setQuick(columnIndex, 1.0)
          }
          rowIndex -> row
        }).asInstanceOf[DrmRdd[Int]]

      val drmInteractions = drmWrap[Int](indexedInteractions, numRows, numColumns)

      IndexedDataset(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        println(this.getClass.toString + ": Schema has illegal values"); throw cce
      }
    }
  }

  private def asOrderedDictionary(entries: Array[String]): BiMap[String, Int] = {
    var dictionary: BiMap[String, Int] = HashBiMap.create()
    var index = 0
    for (entry <- entries) {
      dictionary.forcePut(entry, index)
      index += 1
    }
    dictionary
  }
}

trait TDIndexedDatasetWriter extends Writer[IndexedDataset]{
  /** Read in text delimited tuples from all URIs in this comma delimited source String. 
    *
    * @param mc context for the Spark job
    * @param writeSchema describes the delimiters and positions of values in the output text delimited file.
    * @param dest directory to write text delimited version of [[org.apache.mahout.drivers.IndexedDataset]]
    */
  protected def writer(mc: SparkContext, writeSchema: Schema, dest: String, indexedDataset: IndexedDataset): Unit = {
    try {
      val outDelim1 = writeSchema("delim1").asInstanceOf[String]
      val outDelim2 = writeSchema("delim2").asInstanceOf[String]
      val outDelim3 = writeSchema("delim3").asInstanceOf[String]
      //instance vars must be put into locally scoped vals when put into closures that are
      //executed but Spark
      assert (indexedDataset != null, {println(this.getClass.toString+": has no indexedDataset to write"); throw new IllegalArgumentException })
      assert (!dest.isEmpty, {println(this.getClass.toString+": has no destination or indextedDataset to write"); throw new IllegalArgumentException})
      val matrix: DrmLike[Int] = indexedDataset.matrix
      val rowIDDictionary: BiMap[String, Int] = indexedDataset.rowIDs
      val columnIDDictionary: BiMap[String, Int] = indexedDataset.columnIDs
      // below doesn't compile because the rdd is not in a CheckpointedDrmSpark also I don't know how to turn a
      // CheckpointedDrmSpark[Int] into a DrmLike[Int], which I need to pass in the CooccurrenceAnalysis#cooccurrence
      // This seems to be about the refacotring to abstract away from Spark but the Read and Write are Spark specific
      // and the non-specific DrmLike is no longer attached to a CheckpointedDrmSpark, could be missing something though
      matrix.rdd.map({ case (rowID, itemVector) =>
        var line: String = rowIDDictionary.inverse.get(rowID) + outDelim1
        for (item <- itemVector.nonZeroes()) {
          line += columnIDDictionary.inverse.get(item.index) + outDelim2 + item.get + outDelim3
        }
        line.dropRight(1)
      })
        .saveAsTextFile(dest)
    }catch{
      case cce: ClassCastException => {println(this.getClass.toString+": Schema has illegal values"); throw cce}
    }
  }
}

/** A combined trait that reads and writes */
trait TDIndexedDatasetReaderWriter extends TDIndexedDatasetReader with TDIndexedDatasetWriter

/** Reads text delimited files into an IndexedDataset. Classes are needed to supply trait params in their constructor.
  * @param readSchema describes the delimiters and position of values in the text delimited file to be read.
  * @param mc Spark context for reading files
  * @note The source is supplied by Reader#readFrom .
  * */
class TextDelimitedIndexedDatasetReader(val readSchema: Schema)(implicit val mc: SparkContext) extends TDIndexedDatasetReader

/** Writes  text delimited files into an IndexedDataset. Classes are needed to supply trait params in their constructor.
  * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
  * @param mc Spark context for reading files
  * @note the destination is supplied by Writer#writeTo trait method
  * */
class TextDelimitedIndexedDatasetWriter(val writeSchema: Schema)(implicit val mc: SparkContext) extends TDIndexedDatasetWriter

/** Reads and writes text delimited files to/from an IndexedDataset. Classes are needed to supply trait params in their constructor.
  * @param readSchema describes the delimiters and position of values in the text delimited file(s) to be read.
  * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
  * @param mc Spark context for reading the files, may be implicitly defined.
  * */
class TextDelimitedIndexedDatasetReaderWriter(val readSchema: Schema, val writeSchema: Schema)(implicit val mc: SparkContext) extends TDIndexedDatasetReaderWriter

/** A version of IndexedDataset that has it's own writeTo method from a Writer trait. This is an alternative to creating
  * a Writer based stand-alone class for writing. Consider it experimental allowing similar semantics to drm.writeDrm().
  * Experimental because it's not clear that it is simpler or more intuitive and since IndexedDatasetTextDelimitedWriteables
  * are probably short lived in terms of lines of code so complexity may be moot.
  * @param matrix the data
  * @param rowIDs bi-directional dictionary for rows of external IDs to internal ordinal Mahout IDs.
  * @param columnIDs bi-directional dictionary for columns of external IDs to internal ordinal Mahout IDs.
  * @param writeSchema contains params for the schema/format or the written text delimited file.
  * @param mc mahout distributed context (SparkContext) may be implicitly defined.
  * */
class IndexedDatasetTextDelimitedWriteable(matrix: CheckpointedDrm[Int], rowIDs: BiMap[String,Int], columnIDs: BiMap[String,Int],
                                           val writeSchema: Schema)(implicit val mc: SparkContext)
  extends IndexedDataset(matrix, rowIDs, columnIDs) with TDIndexedDatasetWriter {

  def writeTo(dest: String): Unit = {
    writeTo(this, dest)
  }
}

/**
 * Companion object for the case class [[org.apache.mahout.drivers.IndexedDatasetTextDelimitedWriteable]] primarily used to get a secondary constructor for
 * making one [[org.apache.mahout.drivers.IndexedDatasetTextDelimitedWriteable]] from another. Used when you have a factory like [[org.apache.mahout.drivers.IndexedDatasetStore]]
 * {{{
 * val id = IndexedDatasetTextDelimitedWriteable(indexedDatasetStore.read)
 * }}}
 */

object IndexedDatasetTextDelimitedWriteable {
  /** Secondary constructor for [[org.apache.mahout.drivers.IndexedDataset]] */
  def apply(id2: IndexedDatasetTextDelimitedWriteable) = new IndexedDatasetTextDelimitedWriteable(id2.matrix,  id2.rowIDs, id2.columnIDs, id2.writeSchema)(id2.mc)
}
