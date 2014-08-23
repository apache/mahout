/*
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package org.apache.mahout.drivers

import org.apache.mahout.drivers.RowSimilarityDriver._
import org.apache.mahout.math.cf.CooccurrenceAnalysis
import scala.collection.immutable.HashMap

/**
 * Command line interface for [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]].
 * Reads a text delimited file containing a Mahout DRM of the form
 * (row id, column id: strength, ...). The IDs are user specified strings which will be
 * preserved in the
 * output. The rows define a matrix and [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]]
 * will be used to calculate row-wise self-similarity, or when using two inputs, will
 * calculate cross-similarity between rows of the primary and
 * secondary inputs. Returns one or two directories of text files formatted the same as the input.
 * The options allow control of the input schema, file discovery, output schema, and control of
 * algorithm parameters.
 * To get help run {{{mahout spark-rowsimilarity}}} for a full explanation of options. The default
 * values for formatting will read (rowID<tab>columnID1:strength1<space>columnID2:strength2....)
 * and write (columnID<tab>columnID1:strength1<space>columnID2:strength2....)
 * Each output line will contain a Column ID and similar columns sorted by LLR strength descending.
 * @note To use with a Spark cluster see the --master option, if you run out of heap space check
 *       the --sparkExecutorMemory option.
 */
object RowSimilarityDriver extends MahoutDriver {
  // define only the options specific to RowSimilarity
  private final val RowSimilarityOptions = HashMap[String, Any](
      "maxObservations" -> 500,
      "maxSimilaritiesPerRow" -> 100,
      "appName" -> "RowSimilarityDriver")

  private var readerWriter: TextDelimitedIndexedDatasetReaderWriter = _
  private var readWriteSchema: Schema = _

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutOptionParser(programName = "spark-rowsimilarity") {
      head("spark-rowsimilarity", "Mahout 1.0-SNAPSHOT")

      //Input output options, non-driver specific
      parseIOOptions

      //Algorithm control options--driver specific
      opts = opts ++ RowSimilarityOptions

      note("\nAlgorithm control options:")
      opt[Int]("maxObservations") abbr ("mo") action { (x, options) =>
        options + ("maxObservations" -> x)
      } text ("Max number of observations to consider per row (optional). Default: " +
        RowSimilarityOptions("maxObservations")) validate { x =>
        if (x > 0) success else failure("Option --maxObservations must be > 0")
      }

      opt[Int]('m', "maxSimilaritiesPerRow") action { (x, options) =>
        options + ("maxSimilaritiesPerRow" -> x)
      } text ("Limit the number of similarities per item to this number (optional). Default: " +
        RowSimilarityOptions("maxSimilaritiesPerRow")) validate { x =>
        if (x > 0) success else failure("Option --maxSimilaritiesPerRow must be > 0")
      }

      //Driver notes--driver specific
      note("\nNote: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.")

      //Drm output schema--not driver specific, drm specific
      parseDrmFormatOptions

      //How to search for input
      parseFileDiscoveryOptions

      //Spark config options--not driver specific
      parseSparkOptions

      //Jar inclusion, this option can be set when executing the driver from compiled code, not when from CLI
      parseGenericOptions

      help("help") abbr ("h") text ("prints this usage text\n")

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  override def start(masterUrl: String = parser.opts("master").asInstanceOf[String],
      appName: String = parser.opts("appName").asInstanceOf[String]):
    Unit = {

    // todo: the HashBiMap used in the TextDelimited Reader is hard coded into
    // MahoutKryoRegistrator, it should be added to the register list here so it
    // will be only spcific to this job.
    sparkConf.set("spark.kryo.referenceTracking", "false")
      .set("spark.kryoserializer.buffer.mb", "200")
      .set("spark.executor.memory", parser.opts("sparkExecutorMem").asInstanceOf[String])

    super.start(masterUrl, appName)

    readWriteSchema = new Schema(
        "rowKeyDelim" -> parser.opts("rowKeyDelim").asInstanceOf[String],
        "columnIdStrengthDelim" -> parser.opts("columnIdStrengthDelim").asInstanceOf[String],
        "omitScore" -> parser.opts("omitStrength").asInstanceOf[Boolean],
        "elementDelim" -> parser.opts("elementDelim").asInstanceOf[String])

    readerWriter = new TextDelimitedIndexedDatasetReaderWriter(readWriteSchema, readWriteSchema)

  }

  private def readIndexedDatasets: Array[IndexedDataset] = {

    val inFiles = FileSysUtils(parser.opts("input").asInstanceOf[String], parser.opts("filenamePattern").asInstanceOf[String],
        parser.opts("recursive").asInstanceOf[Boolean]).uris
    val inFiles2 = if (parser.opts("input2") == null || parser.opts("input2").asInstanceOf[String].isEmpty) ""
      else FileSysUtils(parser.opts("input2").asInstanceOf[String], parser.opts("filenamePattern").asInstanceOf[String],
          parser.opts("recursive").asInstanceOf[Boolean]).uris

    if (inFiles.isEmpty) {
      Array()
    } else {

      val datasetA = IndexedDataset(readerWriter.readDRMFrom(inFiles))
      if (parser.opts("writeAllDatasets").asInstanceOf[Boolean]) readerWriter.writeDRMTo(datasetA,
          parser.opts("output").asInstanceOf[String] + "../input-datasets/primary-interactions")

      // The case of readng B can be a bit tricky when the exact same row IDs don't exist for A and B
      // Here we assume there is one row ID space for all interactions. To do this we calculate the
      // row cardinality only after reading in A and B (or potentially C...) We then adjust the
      // cardinality so all match, which is required for the math to work.
      // Note: this may leave blank rows with no representation in any DRM. Blank rows need to
      // be supported (and are at least on Spark) or the row cardinality fix will not work.
      val datasetB = if (!inFiles2.isEmpty) {
        // get cross-cooccurrence interactions from separate files
        val datasetB = IndexedDataset(readerWriter.readDRMFrom(inFiles2, existingRowIDs = datasetA.rowIDs))

        datasetB

      } else {
        null.asInstanceOf[IndexedDataset]
      }
      if (datasetB != null.asInstanceOf[IndexedDataset]) { // do AtB calc
        // true row cardinality is the size of the row id index, which was calculated from all rows of A and B
        val rowCardinality = datasetB.rowIDs.size() // the authoritative row cardinality

        // todo: how expensive is nrow? We could make assumptions about .rowIds that don't rely on
        // its calculation
        val returnedA = if (rowCardinality != datasetA.matrix.nrow) datasetA.newRowCardinality(rowCardinality)
          else datasetA // this guarantees matching cardinality

        val returnedB = if (rowCardinality != datasetB.matrix.nrow) datasetB.newRowCardinality(rowCardinality)
          else datasetB // this guarantees matching cardinality

        if (parser.opts("writeAllDatasets").asInstanceOf[Boolean]) readerWriter.writeDRMTo(datasetB, parser.opts("output") + "../input-datasets/secondary-interactions")

        Array(returnedA, returnedB)
      } else Array(datasetA)
    }
  }

  override def process: Unit = {
    start()

    val indexedDatasets = readIndexedDatasets

    // todo: allow more than one cross-similarity matrix?
    val indicatorMatrices = {
      if (indexedDatasets.length > 1) {
        CooccurrenceAnalysis.cooccurrences(indexedDatasets(0).matrix, parser.opts("randomSeed").asInstanceOf[Int],
            parser.opts("maxSimilaritiesPerRow").asInstanceOf[Int], parser.opts("maxObservations").asInstanceOf[Int],
            Array(indexedDatasets(1).matrix))
      } else {
        CooccurrenceAnalysis.cooccurrences(indexedDatasets(0).matrix, parser.opts("randomSeed").asInstanceOf[Int],
          parser.opts("maxSimilaritiesPerRow").asInstanceOf[Int], parser.opts("maxObservations").asInstanceOf[Int])
      }
    }

    // an alternative is to create a version of IndexedDataset that knows how to write itself
    val selfIndicatorDataset = new IndexedDatasetTextDelimitedWriteable(indicatorMatrices(0), indexedDatasets(0).columnIDs,
      indexedDatasets(0).columnIDs, readWriteSchema)
    selfIndicatorDataset.writeTo(parser.opts("output").asInstanceOf[String] + "indicator-matrix")

    if (indexedDatasets.length > 1) {

      val crossIndicatorDataset = new IndexedDataset(indicatorMatrices(1), indexedDatasets(0).columnIDs, indexedDatasets(1).columnIDs) // cross similarity
      readerWriter.writeDRMTo(crossIndicatorDataset, parser.opts("output").asInstanceOf[String] + "cross-indicator-matrix")

    }

    stop
  }

}
