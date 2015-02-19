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

import org.apache.mahout.math.indexeddataset.{Writer, Reader, Schema, IndexedDataset}
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import org.apache.spark.SparkContext._
import org.apache.mahout.math.RandomAccessSparseVector
import com.google.common.collect.{BiMap, HashBiMap}
import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}
import org.apache.mahout.sparkbindings._
import scala.collection.JavaConversions._

/**
 * Extends Reader trait to supply the [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]] as
 * the type read and a element and row reader functions for reading text delimited files as described in the
 * [[org.apache.mahout.math.indexeddataset.Schema]]
 */
trait TDIndexedDatasetReader extends Reader[IndexedDatasetSpark]{
  /**
   * Read in text delimited elements from all URIs in the comma delimited source String and return
   * the DRM of all elements updating the dictionaries for row and column dictionaries. If there is
   * no strength value in the element, assume it's presence means a strength of 1.
   * @param mc context for the Spark job
   * @param readSchema describes the delimiters and positions of values in the text delimited file.
   * @param source comma delimited URIs of text files to be read from
   * @return a new [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]]
   */
  protected def elementReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create()): IndexedDatasetSpark = {
    try {
      val delimiter = readSchema("delim").asInstanceOf[String]
      val rowIDColumn = readSchema("rowIDColumn").asInstanceOf[Int]
      val columnIDPosition = readSchema("columnIDPosition").asInstanceOf[Int]
      val filterColumn = readSchema("filterColumn").asInstanceOf[Int]
      val filterBy = readSchema("filter").asInstanceOf[String]
      // instance vars must be put into locally scoped vals when used in closures that are executed but Spark

      assert(!source.isEmpty, {
        println(this.getClass.toString + ": has no files to read")
        throw new IllegalArgumentException
      })

      var columns = mc.textFile(source).map { line => line.split(delimiter) }

      // -1 means no filter in the input text, take them all
      if(filterColumn != -1) {
        // get the rows that have a column matching the filter
        columns = columns.filter { tokens => tokens(filterColumn) == filterBy }
      }

      // get row and column IDs
      //val m = columns.collect
      val interactions = columns.map { tokens =>
        tokens(rowIDColumn) -> tokens(columnIDPosition)
      }

      interactions.cache()

      // create separate collections of rowID and columnID tokens
      val rowIDs = interactions.map { case (rowID, _) => rowID }.distinct().collect()
      val columnIDs = interactions.map { case (_, columnID) => columnID }.distinct().collect()

      // create BiMaps for bi-directional lookup of ID by either Mahout ID or external ID
      // broadcast them for access in distributed processes, so they are not recalculated in every task.
      val rowIDDictionary = asOrderedDictionary(existingRowIDs, rowIDs)
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = asOrderedDictionary(entries = columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val ncol = columnIDDictionary.size()
      val nrow = rowIDDictionary.size()

      val indexedInteractions =
        interactions.map { case (rowID, columnID) =>
          val rowIndex = rowIDDictionary_bcast.value.get(rowID).get
          val columnIndex = columnIDDictionary_bcast.value.get(columnID).get

          rowIndex -> columnIndex
        }
        // group by IDs to form row vectors
        .groupByKey().map { case (rowIndex, columnIndexes) =>
          val row = new RandomAccessSparseVector(ncol)
          for (columnIndex <- columnIndexes) {
            row.setQuick(columnIndex, 1.0)
          }
          rowIndex -> row
        }
        .asInstanceOf[DrmRdd[Int]]

      // wrap the DrmRdd and a CheckpointedDrm, which can be used anywhere a DrmLike[Int] is needed
      val drmInteractions = drmWrap[Int](indexedInteractions, nrow, ncol)

      new IndexedDatasetSpark(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        println(this.getClass.toString + ": Schema has illegal values"); throw cce
      }
    }
  }

  /**
   * Read in text delimited rows from all URIs in this comma delimited source String and return
   * the DRM of all elements updating the dictionaries for row and column dictionaries. If there is
   * no strength value in the element, assume it's presence means a strength of 1.
   * @param mc context for the Spark job
   * @param readSchema describes the delimiters and positions of values in the text delimited file.
   * @param source comma delimited URIs of text files to be read into the [[IndexedDatasetSpark]]
   * @return a new [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]]
   */
  protected def rowReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create()): IndexedDatasetSpark = {
    try {
      val rowKeyDelim = readSchema("rowKeyDelim").asInstanceOf[String]
      val columnIdStrengthDelim = readSchema("columnIdStrengthDelim").asInstanceOf[String]
      val elementDelim = readSchema("elementDelim").asInstanceOf[String]
      // no need for omitScore since we can tell if there is a score and assume it is 1.0d if not specified
      //val omitScore = readSchema("omitScore").asInstanceOf[Boolean]

      assert(!source.isEmpty, {
        println(this.getClass.toString + ": has no files to read")
        throw new IllegalArgumentException
      })

      var rows = mc.textFile(source).map { line => line.split(rowKeyDelim) }

      // get row and column IDs
      val interactions = rows.map { row =>
        row(0) -> row(1)// rowID token -> string of column IDs+strengths
      }

      interactions.cache()
      interactions.collect()

      // create separate collections of rowID and columnID tokens
      val rowIDs = interactions.map { case (rowID, _) => rowID }.distinct().collect()

      // the columns are in a TD string so separate them and get unique ones
      val columnIDs = interactions.flatMap { case (_, columns) => columns
        val elements = columns.split(elementDelim)
        val colIDs = elements.map( elem => elem.split(columnIdStrengthDelim)(0) )
        colIDs
      }.distinct().collect()

      // create BiMaps for bi-directional lookup of ID by either Mahout ID or external ID
      // broadcast them for access in distributed processes, so they are not recalculated in every task.
      val rowIDDictionary = asOrderedDictionary(existingRowIDs, rowIDs)
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = asOrderedDictionary(entries = columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val ncol = columnIDDictionary.size()
      val nrow = rowIDDictionary.size()

      val indexedInteractions =
        interactions.map { case (rowID, columns) =>
          val rowIndex = rowIDDictionary_bcast.value.get(rowID).get

          val elements = columns.split(elementDelim)
          val row = new RandomAccessSparseVector(ncol)
          for (element <- elements) {
            val id = element.split(columnIdStrengthDelim)(0)
            val columnID = columnIDDictionary_bcast.value.get(id).get
            val pair = element.split(columnIdStrengthDelim)
            if (pair.size == 2)// there was a strength
              row.setQuick(columnID,pair(1).toDouble)
            else // no strength so set DRM value to 1.0d, this ignores 'omitScore', which is a write param
              row.setQuick(columnID,1.0d)
          }
          rowIndex -> row
        }
        .asInstanceOf[DrmRdd[Int]]

      // wrap the DrmRdd and a CheckpointedDrm, which can be used anywhere a DrmLike[Int] is needed
      val drmInteractions = drmWrap[Int](indexedInteractions, nrow, ncol)

      new IndexedDatasetSpark(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        println(this.getClass.toString + ": Schema has illegal values")
        throw cce
      }
    }
  }

  /**
   * Creates a BiMap from an ID collection. The ID points to an ordinal in which is used internal to Mahout
   * as the row or column ID
   * todo: this is a non-distributed process in an otherwise distributed reader and the BiMap is a
   * non-rdd based object--this will limit the size of the dataset to ones where the dictionaries fit
   * in-memory, the option is to put the dictionaries in rdds and do joins to translate IDs
   */
  private def asOrderedDictionary(dictionary: BiMap[String, Int] = HashBiMap.create(),
      entries: Array[String]):
    BiMap[String, Int] = {
    var index = dictionary.size() // if a dictionary is supplied then add to the end based on the Mahout id 'index'
    for (entry <- entries) {
      if (!dictionary.contains(entry)){
        dictionary.put(entry, index)
        index += 1
      }// the dictionary should never contain an entry since they are supposed to be distinct but for some reason
      // they do
    }
    dictionary
  }
}

/** Extends the Writer trait to supply the type being written and supplies the writer function */
trait TDIndexedDatasetWriter extends Writer[IndexedDatasetSpark]{
  /**
   * Read in text delimited elements from all URIs in this comma delimited source String.
   * @param mc context for the Spark job
   * @param writeSchema describes the delimiters and positions of values in the output text delimited file.
   * @param dest directory to write text delimited version of [[IndexedDatasetSpark]]
   */
  protected def writer(
      mc: DistributedContext,
      writeSchema: Schema,
      dest: String,
      indexedDataset: IndexedDatasetSpark,
      sort: Boolean = true): Unit = {
    try {
      val rowKeyDelim = writeSchema("rowKeyDelim").asInstanceOf[String]
      val columnIdStrengthDelim = writeSchema("columnIdStrengthDelim").asInstanceOf[String]
      val elementDelim = writeSchema("elementDelim").asInstanceOf[String]
      val omitScore = writeSchema("omitScore").asInstanceOf[Boolean]
      //instance vars must be put into locally scoped vals when put into closures that are
      //executed but Spark

      assert(indexedDataset != null, {
        println(this.getClass.toString + ": has no indexedDataset to write")
        throw new IllegalArgumentException
      })
      assert(!dest.isEmpty, {
        println(this.getClass.toString + ": has no destination or indextedDataset to write")
        throw new IllegalArgumentException
      })

      val matrix = indexedDataset.matrix
      val rowIDDictionary = indexedDataset.rowIDs
      val columnIDDictionary = indexedDataset.columnIDs

      matrix.rdd.map { case (rowID, itemVector) =>

        // turn non-zeros into list for sorting
        var itemList = List[(Int, Double)]()
        for (ve <- itemVector.nonZeroes) {
          itemList = itemList :+ (ve.index, ve.get)
        }
        //sort by highest value descending(-)
        val vector = if (sort) itemList.sortBy { elem => -elem._2 } else itemList

        // first get the external rowID token
        if (!vector.isEmpty){
          var line: String = rowIDDictionary.inverse.get(rowID) + rowKeyDelim
          // for the rest of the row, construct the vector contents of elements (external column ID, strength value)
          for (item <- vector) {
            line += columnIDDictionary.inverse.get(item._1)
            if (!omitScore) line += columnIdStrengthDelim + item._2
            line += elementDelim
          }
          // drop the last delimiter, not needed to end the line
          line.dropRight(1)
        } else {//no items so write a line with id but no values, no delimiters
          rowIDDictionary.inverse.get(rowID)
        } // "if" returns a line of text so this must be last in the block
      }
      .saveAsTextFile(dest)

    }catch{
      case cce: ClassCastException => {println(this.getClass.toString+": Schema has illegal values"); throw cce}
    }
  }
}

/** A combined trait that reads and writes */
trait TDIndexedDatasetReaderWriter extends TDIndexedDatasetReader with TDIndexedDatasetWriter

/**
 * Reads text delimited files into an IndexedDataset. Classes can be used to supply trait params in their constructor.
 * @param readSchema describes the delimiters and position of values in the text delimited file to be read.
 * @param mc Spark context for reading files
 * @note The source is supplied to Reader#readElementsFrom .
 */
class TextDelimitedIndexedDatasetReader(val readSchema: Schema)
    (implicit val mc: DistributedContext) extends TDIndexedDatasetReader

/**
 * Writes  text delimited files into an IndexedDataset. Classes can be used to supply trait params in their
 * constructor.
 * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
 * @param mc Spark context for reading files
 * @note the destination is supplied to Writer#writeTo
 */
class TextDelimitedIndexedDatasetWriter(val writeSchema: Schema, val sort: Boolean = true)
    (implicit val mc: DistributedContext)
  extends TDIndexedDatasetWriter

/**
 * Reads and writes text delimited files to/from an IndexedDataset. Classes are needed to supply trait params in
 * their constructor.
 * @param readSchema describes the delimiters and position of values in the text delimited file(s) to be read.
 * @param writeSchema describes the delimiters and position of values in the text delimited file(s) written.
 * @param mc Spark context for reading the files, may be implicitly defined.
 */
class TextDelimitedIndexedDatasetReaderWriter(val readSchema: Schema, val writeSchema: Schema, val sort: Boolean = true)
    (implicit val mc: DistributedContext)
  extends TDIndexedDatasetReaderWriter

