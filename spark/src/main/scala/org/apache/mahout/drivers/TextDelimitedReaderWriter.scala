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

import scala.collection.JavaConversions._
import org.apache.spark.SparkContext._
import org.apache.mahout.math.RandomAccessSparseVector
import com.google.common.collect.{BiMap, HashBiMap}
import scala.collection.JavaConversions._
import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}
import org.apache.mahout.sparkbindings._


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
  protected def tupleReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create()): IndexedDataset = {
    try {
      val delimiter = readSchema("delim").asInstanceOf[String]
      val rowIDPosition = readSchema("rowIDPosition").asInstanceOf[Int]
      val columnIDPosition = readSchema("columnIDPosition").asInstanceOf[Int]
      val filterPosition = readSchema("filterPosition").asInstanceOf[Int]
      val filterBy = readSchema("filter").asInstanceOf[String]
      // instance vars must be put into locally scoped vals when used in closures that are executed but Spark

      assert(!source.isEmpty, {
        println(this.getClass.toString + ": has no files to read")
        throw new IllegalArgumentException
      })

      var columns = mc.textFile(source).map { line => line.split(delimiter) }
      val m = columns.collect

      // -1 means no filter in the input text, take them all
      if(filterPosition != -1) {
        // get the rows that have a column matching the filter
        columns = columns.filter { tokens => tokens(filterPosition) == filterBy }
      }

      // get row and column IDs
      //val m = columns.collect
      val interactions = columns.map { tokens =>
        tokens(rowIDPosition) -> tokens(columnIDPosition)
      }

      interactions.cache()

      // create separate collections of rowID and columnID tokens
      val rowIDs = interactions.map { case (rowID, _) => rowID }.distinct().collect()
      val columnIDs = interactions.map { case (_, columnID) => columnID }.distinct().collect()

      val numRows = rowIDs.size
      val numColumns = columnIDs.size

      // create BiMaps for bi-directional lookup of ID by either Mahout ID or external ID
      // broadcast them for access in distributed processes, so they are not recalculated in every task.
      val rowIDDictionary = asOrderedDictionary(existingRowIDs, rowIDs)
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = asOrderedDictionary(entries = columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val indexedInteractions =
        interactions.map { case (rowID, columnID) =>
          val rowIndex = rowIDDictionary_bcast.value.get(rowID).get
          val columnIndex = columnIDDictionary_bcast.value.get(columnID).get

          rowIndex -> columnIndex
        }
        // group by IDs to form row vectors
        .groupByKey().map { case (rowIndex, columnIndexes) =>
          val row = new RandomAccessSparseVector(numColumns)
          for (columnIndex <- columnIndexes) {
            row.setQuick(columnIndex, 1.0)
          }
          rowIndex -> row
        }
        .asInstanceOf[DrmRdd[Int]]

      // wrap the DrmRdd and a CheckpointedDrm, which can be used anywhere a DrmLike[Int] is needed
      val drmInteractions = drmWrap[Int](indexedInteractions, numRows, numColumns)

      IndexedDataset(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        println(this.getClass.toString + ": Schema has illegal values"); throw cce
      }
    }
  }

  // this creates a BiMap from an ID collection. The ID points to an ordinal int
  // which is used internal to Mahout as the row or column ID
  // todo: this is a non-distributed process and the BiMap is a non-rdd based object--might be a scaling problem
  private def asOrderedDictionary(dictionary: BiMap[String, Int] = HashBiMap.create(), entries: Array[String]): BiMap[String, Int] = {
    var index = dictionary.size() // if a dictionary is supplied then add to the end based on the Mahout id 'index'
    for (entry <- entries) {
      if (!dictionary.contains(entry)) dictionary.put(entry, index)
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
  protected def writer(mc: DistributedContext, writeSchema: Schema, dest: String, indexedDataset: IndexedDataset): Unit = {
    try {
      val rowKeyDelim = writeSchema("rowKeyDelim").asInstanceOf[String]
      val columnIdStrengthDelim = writeSchema("columnIdStrengthDelim").asInstanceOf[String]
      val tupleDelim = writeSchema("tupleDelim").asInstanceOf[String]
      val omitScore = writeSchema("omitScore").asInstanceOf[Boolean]
      //instance vars must be put into locally scoped vals when put into closures that are
      //executed but Spark

      assert(indexedDataset != null, {
        println(this.getClass.toString + ": has no indexedDataset to write"); throw new IllegalArgumentException
      })
      assert(!dest.isEmpty, {
        println(this.getClass.toString + ": has no destination or indextedDataset to write"); throw new IllegalArgumentException
      })

      val matrix = indexedDataset.matrix
      val rowIDDictionary = indexedDataset.rowIDs
      val columnIDDictionary = indexedDataset.columnIDs

      matrix.rdd.map { case (rowID, itemVector) =>

        // each line is created of non-zero values with schema specified delimiters and original row and column ID tokens
        // first get the external rowID token
        var line: String = rowIDDictionary.inverse.get(rowID) + rowKeyDelim

        // for the rest of the row, construct the vector contents of tuples (external column ID, strength value)
        for (item <- itemVector.nonZeroes()) {
          line += columnIDDictionary.inverse.get(item.index)
          if (!omitScore) line += columnIdStrengthDelim + item.get
          line += tupleDelim
        }
        // drop the last delimiter, not needed to end the line
        line.dropRight(1)
      }
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
  * @note The source is supplied by Reader#readTuplesFrom .
  * */
class TextDelimitedIndexedDatasetReader(val readSchema: Schema)(implicit val mc: DistributedContext) extends TDIndexedDatasetReader

/** Writes  text delimited files into an IndexedDataset. Classes are needed to supply trait params in their constructor.
  * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
  * @param mc Spark context for reading files
  * @note the destination is supplied by Writer#writeDRMTo trait method
  * */
class TextDelimitedIndexedDatasetWriter(val writeSchema: Schema)(implicit val mc: DistributedContext) extends TDIndexedDatasetWriter

/** Reads and writes text delimited files to/from an IndexedDataset. Classes are needed to supply trait params in their constructor.
  * @param readSchema describes the delimiters and position of values in the text delimited file(s) to be read.
  * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
  * @param mc Spark context for reading the files, may be implicitly defined.
  * */
class TextDelimitedIndexedDatasetReaderWriter(val readSchema: Schema, val writeSchema: Schema)(implicit val mc: DistributedContext) extends TDIndexedDatasetReaderWriter

/** A version of IndexedDataset that has it's own writeDRMTo method from a Writer trait. This is an alternative to creating
  * a Writer based stand-alone class for writing. Consider it experimental allowing similar semantics to drm.writeDrm().
  * Experimental because it's not clear that it is simpler or more intuitive and since IndexedDatasetTextDelimitedWriteables
  * are probably short lived in terms of lines of code so complexity may be moot.
  * @param matrix the data
  * @param rowIDs bi-directional dictionary for rows of external IDs to internal ordinal Mahout IDs.
  * @param columnIDs bi-directional dictionary for columns of external IDs to internal ordinal Mahout IDs.
  * @param writeSchema contains params for the schema/format or the written text delimited file.
  * @param mc mahout distributed context (DistributedContext) may be implicitly defined.
  * */
class IndexedDatasetTextDelimitedWriteable(matrix: CheckpointedDrm[Int], rowIDs: BiMap[String,Int], columnIDs: BiMap[String,Int],
                                           val writeSchema: Schema)(implicit val mc: DistributedContext)
  extends IndexedDataset(matrix, rowIDs, columnIDs) with TDIndexedDatasetWriter {

  def writeTo(dest: String): Unit = {
    writeDRMTo(this, dest)
  }
}

/**
 * Companion object for the case class [[org.apache.mahout.drivers.IndexedDatasetTextDelimitedWriteable]] primarily used to get a secondary constructor for
 * making one [[org.apache.mahout.drivers.IndexedDatasetTextDelimitedWriteable]] from another. Used when you have a factory like [[org.apache.mahout.drivers.TextDelimitedIndexedDatasetReader]]
 * {{{
 *   val id = IndexedDatasetTextDelimitedWriteable(indexedDatasetReader.readTuplesFrom(source))
 * }}}
 */

object IndexedDatasetTextDelimitedWriteable {
  /** Secondary constructor for [[org.apache.mahout.drivers.IndexedDataset]] */
  def apply(id2: IndexedDatasetTextDelimitedWriteable) = new IndexedDatasetTextDelimitedWriteable(id2.matrix,  id2.rowIDs, id2.columnIDs, id2.writeSchema)(id2.mc)
}
