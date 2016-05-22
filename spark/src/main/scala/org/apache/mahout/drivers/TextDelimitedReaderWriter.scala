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

import org.apache.log4j.Logger
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.indexeddataset._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark

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
    existingRowIDs: Option[BiDictionary] = None): IndexedDatasetSpark = {
    @transient lazy val logger = Logger.getLogger(this.getClass.getCanonicalName)
    try {
      val delimiter = readSchema("delim").asInstanceOf[String]
      val rowIDColumn = readSchema("rowIDColumn").asInstanceOf[Int]
      val columnIDPosition = readSchema("columnIDPosition").asInstanceOf[Int]
      val filterColumn = readSchema("filterColumn").asInstanceOf[Int]
      val filterBy = readSchema("filter").asInstanceOf[String]
      // instance vars must be put into locally scoped vals when used in closures that are executed but Spark


      require (!source.isEmpty, "No file(s) to read")

      var columns = mc.textFile(source).map { line => line.split(delimiter) }

      // -1 means no filter in the input text, take them all
      if(filterColumn != -1) {
        // get the rows that have a column matching the filter
        columns = columns.filter { tokens => tokens(filterColumn) == filterBy }
      }

      // get row and column IDs
      val interactions = columns.map { tokens =>
        tokens(rowIDColumn) -> tokens(columnIDPosition)
      }

      interactions.cache()

      // create separate collections of rowID and columnID tokens
      val rowIDs = interactions.map { case (rowID, _) => rowID }.distinct().collect()
      val columnIDs = interactions.map { case (_, columnID) => columnID }.distinct().collect()

      // create BiDictionary(s) for bi-directional lookup of ID by either Mahout ID or external ID
      // broadcast them for access in distributed processes, so they are not recalculated in every task.
      //val rowIDDictionary = BiDictionary.append(existingRowIDs, rowIDs)
      val rowIDDictionary = existingRowIDs match {
        case Some(d) => d.merge(rowIDs)
        case None =>  new BiDictionary(rowIDs)
      }
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = new BiDictionary(keys = columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val ncol = columnIDDictionary.size
      val nrow = rowIDDictionary.size

      val indexedInteractions =
        interactions.map { case (rowID, columnID) =>
          val rowIndex = rowIDDictionary_bcast.value.getOrElse(rowID, -1)
          val columnIndex = columnIDDictionary_bcast.value.getOrElse(columnID, -1)

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
      val drmInteractions = drmWrap[Int](indexedInteractions)

      new IndexedDatasetSpark(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        logger.error("Schema has illegal values"); throw cce
      }
    }
  }

  /**
   * Read in text delimited rows from all URIs in this comma delimited source String and return
   * the DRM of all elements updating the dictionaries for row and column dictionaries. If there is
   * no strength value in the element, assume it's presence means a strength of 1.
   * Note: if the input file has a strength delimiter but none is seen in rows, we assume there is none
   *   and give the strength as 1 in the input DRM.
   * @param mc context for the Spark job
   * @param readSchema describes the delimiters and positions of values in the text delimited file.
   * @param source comma delimited URIs of text files to be read into the [[IndexedDatasetSpark]]
   * @return a new [[org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark]]
   */
  protected def rowReader(
    mc: DistributedContext,
    readSchema: Schema,
    source: String,
    existingRowIDs: Option[BiDictionary] = None): IndexedDatasetSpark = {
    @transient lazy val logger = Logger.getLogger(this.getClass.getCanonicalName)
    try {
      val rowKeyDelim = readSchema("rowKeyDelim").asInstanceOf[String]
      val columnIdStrengthDelim = readSchema("columnIdStrengthDelim").asInstanceOf[String]
      val elementDelim = readSchema("elementDelim").asInstanceOf[String]
      val omitScore = readSchema("omitScore").asInstanceOf[Boolean]

      require (!source.isEmpty, "No file(s) to read")
      val rows = mc.textFile(source).map { line => line.split(rowKeyDelim) }

      // get row and column IDs
      val interactions = rows.map { row =>
        // rowID token -> string of column IDs+strengths or null if empty (all elements zero)
        row(0) -> (if (row.length > 1) row(1) else null)
      }

      interactions.cache()

      // create separate collections of rowID and columnID tokens
      val rowIDs = interactions.map { case (rowID, _) => rowID }.distinct().collect()

      // the columns are in a TD string so separate them and get unique ones
      val columnIDs = interactions.flatMap { case (_, columns) => columns
        if (columns == null) None
        else {
          val elements = columns.split(elementDelim)
          val colIDs = if (!omitScore)
            elements.map(elem => elem.split(columnIdStrengthDelim)(0))
          else
            elements
          colIDs
        }
      }.distinct().collect()

      // create BiMaps for bi-directional lookup of ID by either Mahout ID or external ID
      // broadcast them for access in distributed processes, so they are not recalculated in every task.
      //val rowIDDictionary = BiDictionary.append(existingRowIDs, rowIDs)
      val rowIDDictionary = existingRowIDs match {
        case Some(d) => d.merge(rowIDs)
        case None =>  new BiDictionary(rowIDs)
      }
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = new BiDictionary(keys = columnIDs)
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      val ncol = columnIDDictionary.size
      val nrow = rowIDDictionary.size

      val indexedInteractions =
        interactions.map { case (rowID, columns) =>
          val rowIndex = rowIDDictionary_bcast.value.getOrElse(rowID, -1)

          val row = new RandomAccessSparseVector(ncol)
          if (columns != null) {
            val elements = columns.split(elementDelim)
            for (element <- elements) {
              val id = if (omitScore) element else element.split(columnIdStrengthDelim)(0)
              val columnID = columnIDDictionary_bcast.value.getOrElse(id, -1)
              val strength = if (omitScore) 1.0d
              else {
                // if the input says not to omit but there is no seperator treat
                // as omitting and return a strength of 1
                if (element.split(columnIdStrengthDelim).size == 1) 1.0d
                else element.split(columnIdStrengthDelim)(1).toDouble
              }
              row.setQuick(columnID, strength)
            }
          }
          rowIndex -> row
        }
        .asInstanceOf[DrmRdd[Int]]

      // wrap the DrmRdd in a CheckpointedDrm, which can be used anywhere a DrmLike[Int] is needed
      val drmInteractions = drmWrap[Int](indexedInteractions)

      new IndexedDatasetSpark(drmInteractions, rowIDDictionary, columnIDDictionary)

    } catch {
      case cce: ClassCastException => {
        logger.error("Schema has illegal values")
        throw cce
      }
    }
  }

  /**
   * Creates a BiDictionary from an ID collection. The ID points to an ordinal in which is used internal to Mahout
   * as the row or column ID
   * todo: this is a non-distributed process in an otherwise distributed reader and the BiDictionary is a
   * non-rdd based object--this will limit the size of the dataset to ones where the dictionaries fit
   * in-memory, the option is to put the dictionaries in rdds and do joins to translate IDs
   */
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
    @transient lazy val logger = Logger.getLogger(this.getClass.getCanonicalName)
    try {
      val rowKeyDelim = writeSchema("rowKeyDelim").asInstanceOf[String]
      val columnIdStrengthDelim = writeSchema("columnIdStrengthDelim").asInstanceOf[String]
      val elementDelim = writeSchema("elementDelim").asInstanceOf[String]
      val omitScore = writeSchema("omitScore").asInstanceOf[Boolean]
      //instance vars must be put into locally scoped vals when put into closures that are
      //executed but Spark

      require (indexedDataset != null ,"No IndexedDataset to write")
      require (!dest.isEmpty,"No destination to write to")

      val matrix = indexedDataset.matrix.checkpoint()
      val rowIDDictionary = indexedDataset.rowIDs
      val rowIDDictionary_bcast = mc.broadcast(rowIDDictionary)

      val columnIDDictionary = indexedDataset.columnIDs
      val columnIDDictionary_bcast = mc.broadcast(columnIDDictionary)

      matrix.rdd.map { case (rowID, itemVector) =>

        // turn non-zeros into list for sorting
        var itemList = List[(Int, Double)]()
        for (ve <- itemVector.nonZeroes) {
          itemList = itemList :+ (ve.index, ve.get)
        }
        //sort by highest value descending(-)
        val vector = if (sort) itemList.sortBy { elem => -elem._2 } else itemList

        // first get the external rowID token
        if (vector.nonEmpty){
          var line = rowIDDictionary_bcast.value.inverse.getOrElse(rowID, "INVALID_ROW_ID") + rowKeyDelim
          // for the rest of the row, construct the vector contents of elements (external column ID, strength value)
          for (item <- vector) {
            line += columnIDDictionary_bcast.value.inverse.getOrElse(item._1, "INVALID_COLUMN_ID")
            if (!omitScore) line += columnIdStrengthDelim + item._2
            line += elementDelim
          }
          // drop the last delimiter, not needed to end the line
          line.dropRight(1)
        } else {//no items so write a line with id but no values, no delimiters
          rowIDDictionary_bcast.value.inverse.getOrElse(rowID, "INVALID_ROW_ID")
        } // "if" returns a line of text so this must be last in the block
      }
      .saveAsTextFile(dest)

    }catch{
      case cce: ClassCastException => {
        logger.error("Schema has illegal values"); throw cce}
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

