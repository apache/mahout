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
import org.apache.mahout.math.indexeddataset.{Schema, IndexedDataset, indexedDatasetDFSReadElements}
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import scala.collection.immutable.HashMap

/**
 * Command line interface for [[org.apache.mahout.math.cf.SimilarityAnalysis#cooccurrencesIDSs]]. Reads text lines
 * that contain (row id, column id, ...). The IDs are user specified strings which will be preserved in the output.
 * The individual elements will be accumulated into a matrix like
 * [[org.apache.mahout.math.indexeddataset.IndexedDataset]] and
 * [[org.apache.mahout.math.cf.SimilarityAnalysis#cooccurrencesIDSs]] will be used to calculate row-wise
 * self-similarity, or when using filters or two inputs, will generate two matrices and calculate both the
 * self-similarity of the primary matrix and the row-wise similarity of the primary to the secondary. Returns one
 * or two directories of text files formatted as specified in the options. The options allow flexible control of the
 * input schema, file discovery, output schema, and control of algorithm parameters. To get help run
 * {{{mahout spark-itemsimilarity}}} for a full explanation of options. To process simple elements of text delimited
 * values (userID,itemID) with or without a strengths and with a separator of tab, comma, or space, you can specify
 * only the input and output file and directory--all else will default to the correct values. Each output line will
 * contain the Item ID and similar items sorted by LLR strength descending.
 * @note To use with a Spark cluster see the --master option, if you run out of heap space check
 *       the --sparkExecutorMemory option. Other [[org.apache.spark.SparkConf]] key value pairs can be with the -D:k=v
 *       option.
 */
object ItemSimilarityDriver extends MahoutSparkDriver {
  // define only the options specific to ItemSimilarity
  private final val ItemSimilarityOptions = HashMap[String, Any](
    "maxPrefs" -> 500,
    "maxSimilaritiesPerItem" -> 100,
    "appName" -> "ItemSimilarityDriver")

  private var writeSchema: Schema = _
  private var readSchema1: Schema = _
  private var readSchema2: Schema = _

  /**
   * Entry point, not using Scala App trait
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-itemsimilarity") {
      head("spark-itemsimilarity", "Mahout")

      //Input output options, non-driver specific
      parseIOOptions(numInputs = 2)

      //Algorithm control options--driver specific
      opts = opts ++ ItemSimilarityOptions
      note("\nAlgorithm control options:")
      opt[Int]("maxPrefs") abbr ("mppu") action { (x, options) =>
        options + ("maxPrefs" -> x)
      } text ("Max number of preferences to consider per user (optional). Default: " +
        ItemSimilarityOptions("maxPrefs")) validate { x =>
          if (x > 0) success else failure("Option --maxPrefs must be > 0")
      }

      // not implemented in SimilarityAnalysis.cooccurrence
      // threshold, and minPrefs
      // todo: replacing the threshold with some % of the best values and/or a
      // confidence measure expressed in standard deviations would be nice.

      opt[Int]('m', "maxSimilaritiesPerItem") action { (x, options) =>
        options + ("maxSimilaritiesPerItem" -> x)
      } text ("Limit the number of similarities per item to this number (optional). Default: " +
        ItemSimilarityOptions("maxSimilaritiesPerItem")) validate { x =>
          if (x > 0) success else failure("Option --maxSimilaritiesPerItem must be > 0")
      }

      //Driver notes--driver specific
      note("\nNote: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.")

      //Input text format
      parseElementInputSchemaOptions()

      //How to search for input
      parseFileDiscoveryOptions()

      //Drm output schema--not driver specific, drm specific
      parseIndexedDatasetFormatOptions()

      //Spark config options--not driver specific
      parseSparkOptions()

      //Jar inclusion, this option can be set when executing the driver from compiled code, not when from CLI
      parseGenericOptions()

      help("help") abbr ("h") text ("prints this usage text\n")

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process()
    }
  }

  override protected def start() : Unit = {

    super.start()

    readSchema1 = new Schema("delim" -> parser.opts("inDelim").asInstanceOf[String],
      "filter" -> parser.opts("filter1").asInstanceOf[String],
      "rowIDColumn" -> parser.opts("rowIDColumn").asInstanceOf[Int],
      "columnIDPosition" -> parser.opts("itemIDColumn").asInstanceOf[Int],
      "filterColumn" -> parser.opts("filterColumn").asInstanceOf[Int])

    if ((parser.opts("filterColumn").asInstanceOf[Int] != -1 && parser.opts("filter2").asInstanceOf[String] != null)
      || (parser.opts("input2").asInstanceOf[String] != null && !parser.opts("input2").asInstanceOf[String].isEmpty )){
      // only need to change the filter used compared to readSchema1
      readSchema2 = new Schema(readSchema1) += ("filter" -> parser.opts("filter2").asInstanceOf[String])

    }

    writeSchema = new Schema(
      "rowKeyDelim" -> parser.opts("rowKeyDelim").asInstanceOf[String],
      "columnIdStrengthDelim" -> parser.opts("columnIdStrengthDelim").asInstanceOf[String],
      "omitScore" -> parser.opts("omitStrength").asInstanceOf[Boolean],
      "elementDelim" -> parser.opts("elementDelim").asInstanceOf[String])
  }

  private def readIndexedDatasets: Array[IndexedDataset] = {

    val inFiles = HDFSPathSearch(parser.opts("input").asInstanceOf[String],
      parser.opts("filenamePattern").asInstanceOf[String], parser.opts("recursive").asInstanceOf[Boolean]).uris
    val inFiles2 = if (parser.opts("input2") == null || parser.opts("input2").asInstanceOf[String].isEmpty) ""
    else HDFSPathSearch(parser.opts("input2").asInstanceOf[String], parser.opts("filenamePattern").asInstanceOf[String],
      parser.opts("recursive").asInstanceOf[Boolean]).uris

    if (inFiles.isEmpty) {
      Array()
    } else {

      val datasetA = indexedDatasetDFSReadElements(inFiles,readSchema1)
      if (parser.opts("writeAllDatasets").asInstanceOf[Boolean])
        datasetA.dfsWrite(parser.opts("output").asInstanceOf[String] + "../input-datasets/primary-interactions",
          schema = writeSchema)

      // The case of reading B can be a bit tricky when the exact same row IDs don't exist for A and B
      // Here we assume there is one row ID space for all interactions. To do this we calculate the
      // row cardinality only after reading in A and B (or potentially C...) We then adjust the cardinality
      // so all match, which is required for the math to work.
      // Note: this may leave blank rows with no representation in any DRM. Blank rows need to
      // be supported (and are at least on Spark) or the row cardinality adjustment will not work.
      val datasetB = if (!inFiles2.isEmpty) {
        // get cross-cooccurrence interactions from separate files
        val datasetB = indexedDatasetDFSReadElements(inFiles2, readSchema2, existingRowIDs = Some(datasetA.rowIDs))

        datasetB

      } else if (parser.opts("filterColumn").asInstanceOf[Int] != -1
        && parser.opts("filter2").asInstanceOf[String] != null) {

        // get cross-cooccurrences interactions by using two filters on a single set of files
        val datasetB = indexedDatasetDFSReadElements(inFiles, readSchema2, existingRowIDs = Some(datasetA.rowIDs))

        datasetB

      } else {
        null.asInstanceOf[IndexedDatasetSpark]
      }
      if (datasetB != null.asInstanceOf[IndexedDataset]) { // do AtB calc
        // true row cardinality is the size of the row id index, which was calculated from all rows of A and B
        val rowCardinality = datasetB.rowIDs.size // the authoritative row cardinality

        val returnedA = if (rowCardinality != datasetA.matrix.nrow) datasetA.newRowCardinality(rowCardinality)
        else datasetA // this guarantees matching cardinality

        val returnedB = if (rowCardinality != datasetB.matrix.nrow) datasetB.newRowCardinality(rowCardinality)
        else datasetB // this guarantees matching cardinality

        if (parser.opts("writeAllDatasets").asInstanceOf[Boolean])
          datasetB.dfsWrite(parser.opts("output").asInstanceOf[String] + "../input-datasets/secondary-interactions",
            schema = writeSchema)
        Array(returnedA, returnedB)
      } else Array(datasetA)
    }
  }

  override def process(): Unit = {
    start()

    val indexedDatasets = readIndexedDatasets
    val idss = SimilarityAnalysis.cooccurrencesIDSs(indexedDatasets, parser.opts("randomSeed").asInstanceOf[Int],
      parser.opts("maxSimilaritiesPerItem").asInstanceOf[Int], parser.opts("maxPrefs").asInstanceOf[Int])

    // todo: allow more than one cross-similarity matrix?
    idss(0).dfsWrite(parser.opts("output").asInstanceOf[String] + "similarity-matrix", schema = writeSchema)
    if(idss.length > 1)
      idss(1).dfsWrite(parser.opts("output").asInstanceOf[String] + "cross-similarity-matrix", schema = writeSchema)

    stop()
  }

}
