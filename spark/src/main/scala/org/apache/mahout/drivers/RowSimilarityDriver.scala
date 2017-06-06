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

import org.apache.mahout.common.HDFSPathSearch
import org.apache.mahout.math.cf.SimilarityAnalysis
import org.apache.mahout.math.indexeddataset.{Schema, IndexedDataset, indexedDatasetDFSRead}
import scala.collection.immutable.HashMap

/**
 * Command line interface for [[org.apache.mahout.math.cf.SimilarityAnalysis#rowSimilarityIDSs( )]].
 * Reads a text delimited file containing rows of a [[org.apache.mahout.math.indexeddataset.IndexedDataset]]
 * with domain specific IDS of the form (row id, column id: strength, ...). The IDs will be preserved in the
 * output. The rows define a matrix and [[org.apache.mahout.math.cf.SimilarityAnalysis#rowSimilarityIDSs( )]]
 * will be used to calculate row-wise similarity using log-likelihood. The options allow control of the input
 * schema, file discovery, output schema, and control of algorithm parameters.
 *
 * To get help run {{{mahout spark-rowsimilarity}}} for a full explanation of options. The default
 * values for formatting will read (rowID<tab>columnID1:strength1<space>columnID2:strength2....)
 * and write (rowID<tab>rowID1:strength1<space>rowID2:strength2....)
 * Each output line will contain a row ID and similar columns sorted by LLR strength descending.
 * @note To use with a Spark cluster see the --master option, if you run out of heap space check
 *       the --sparkExecutorMemory option.
 */
object RowSimilarityDriver extends MahoutSparkDriver {
  // define only the options specific to RowSimilarity
  private final val RowSimilarityOptions = HashMap[String, Any](
    "maxObservations" -> 500,
    "maxSimilaritiesPerRow" -> 100,
    "appName" -> "RowSimilarityDriver")

  private var readWriteSchema: Schema = _

  /**
   * Entry point, not using Scala App trait
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-rowsimilarity") {
      head("spark-rowsimilarity", "Mahout")

      //Input output options, non-driver specific
      parseIOOptions()

      //Algorithm control options--driver specific
      opts = opts ++ RowSimilarityOptions

      note("\nAlgorithm control options:")
      opt[Int]("maxObservations") abbr "mo" action { (x, options) =>
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

      // --threshold not implemented in SimilarityAnalysis.rowSimilarity
      // todo: replacing the threshold with some % of the best values and/or a
      // confidence measure expressed in standard deviations would be nice.

      //Driver notes--driver specific
      note("\nNote: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.")

      //Drm output schema--not driver specific, drm specific
      parseIndexedDatasetFormatOptions("\nInput and Output text file schema options (same for both):")

      //How to search for input
      parseFileDiscoveryOptions()

      //Spark config options--not driver specific
      parseSparkOptions()

      //Jar inclusion, this option can be set when executing the driver from compiled code, not when from CLI
      parseGenericOptions()

      help("help") abbr "h" text "prints this usage text\n"

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  override protected def start(): Unit = {

    super.start()

    readWriteSchema = new Schema(
      "rowKeyDelim" -> parser.opts("rowKeyDelim").asInstanceOf[String],
      "columnIdStrengthDelim" -> parser.opts("columnIdStrengthDelim").asInstanceOf[String],
      "omitScore" -> parser.opts("omitStrength").asInstanceOf[Boolean],
      "elementDelim" -> parser.opts("elementDelim").asInstanceOf[String])

  }

  private def readIndexedDataset: IndexedDataset = {

    val inFiles = HDFSPathSearch(parser.opts("input").asInstanceOf[String],
      parser.opts("filenamePattern").asInstanceOf[String], parser.opts("recursive").asInstanceOf[Boolean]).uris

    if (inFiles.isEmpty) {
      null.asInstanceOf[IndexedDataset]
    } else {

      val datasetA = indexedDatasetDFSRead(src = inFiles, schema = readWriteSchema)
      datasetA
    }
  }

  override def process(): Unit = {
    start()

    val indexedDataset = readIndexedDataset

    val rowSimilarityIDS = SimilarityAnalysis.rowSimilarityIDS(indexedDataset,
      parser.opts("randomSeed").asInstanceOf[Int],
      parser.opts("maxSimilaritiesPerRow").asInstanceOf[Int],
      parser.opts("maxObservations").asInstanceOf[Int])

    rowSimilarityIDS.dfsWrite(dest = parser.opts("output").asInstanceOf[String], schema = readWriteSchema)

    stop()
  }

}
