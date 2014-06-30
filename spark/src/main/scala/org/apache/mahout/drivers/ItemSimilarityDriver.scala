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

import org.apache.mahout.cf.CooccurrenceAnalysis

/**
 * Command line interface for [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]].
 * Command line interface for [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]].
 * It reads text lines
 * that contain (row id, column id, ...). The IDs are user specified strings which will be
 * preserved in the
 * output. The individual tuples will be accumulated into a matrix and [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]]
 * will be used to calculate row-wise self-similarity, or when using filters, will generate two
 * matrices and calculate both the self similarity of the primary matrix and the row-wise
 * similarity of the primary
 * to the secondary. Returns one or two directories of text files formatted as specified in
 * the options.
 */
object ItemSimilarityDriver extends MahoutDriver {
  //todo: Should also take two input streams and do cross similarity with no filter required.
  // required for examples

  private var options: Options = _
  private var reader1: TextDelimitedIndexedDatasetReader = _
  private var reader2: TextDelimitedIndexedDatasetReader = _
  private var writer: TextDelimitedIndexedDatasetWriter = _
  private var writeSchema: Schema = _

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {
    val parser = new MahoutOptionParser[Options]("spark-itemsimilarity") {
      head("spark-itemsimilarity", "Mahout 1.0-SNAPSHOT")

      //Input output options, non-driver specific
      note("Input, output options")
      opt[String]('i', "input") required() action { (x, options) =>
        options.copy(input = x)
      } text ("Input path, may be a filename, directory name, or comma delimited list of HDFS supported URIs (required)")

      opt[String]('o', "output") required() action { (x, options) =>
        if (x.endsWith("/")) // todo: check to see if HDFS allows MS-Windows backslashes locally?
          options.copy(output = x)
        else
          options.copy(output = x + "/")
      } text ("Path for output, any local or HDFS supported URI (required).")

      //Algorithm control options--driver specific
      note("\nAlgorithm control options:")
      opt[String]("master") abbr ("ma") text ("Spark Master URL (optional). Default: \"local\". Note that you can specify the number of cores to get a performance improvement, for example \"local[4]\"") action { (x, options) =>
        options.copy(master = x)
      }

      opt[Int]("maxPrefs") abbr ("mppu") action { (x, options) =>
        options.copy(maxPrefs = x)
      } text ("Max number of preferences to consider per user (optional). Default: 500") validate { x =>
        if (x > 0) success else failure("Option --maxPrefs must be > 0")
      }

/** not implemented in CooccurrenceAnalysis.cooccurrence
      opt[Int]("minPrefs") abbr ("mp") action { (x, options) =>
        options.copy(minPrefs = x)
      } text ("Ignore users with less preferences than this (optional). Default: 1") validate { x =>
        if (x > 0) success else failure("Option --minPrefs must be > 0")
      }
*/

      opt[Int]('m', "maxSimilaritiesPerItem") action { (x, options) =>
        options.copy(maxSimilaritiesPerItem = x)
      } text ("Limit the number of similarities per item to this number (optional). Default: 100") validate { x =>
        if (x > 0) success else failure("Option --maxSimilaritiesPerItem must be > 0")
      }

      opt[Int]("randomSeed") abbr ("rs") action { (x, options) =>
        options.copy(randomSeed = x)
      } text ("Int to seed random number generator (optional). Default: Uses time to generate a seed") validate { x =>
        if (x > 0) success else failure("Option --randomSeed must be > 0")
      }

      //Input text file schema--not driver specific but input data specific, tuples input,
      // not drms
      note("\nInput text file schema options:")
      opt[String]("inDelim") abbr ("id") text ("Input delimiter character (optional). Default: \"[,\\t]\"") action { (x, options) =>
        options.copy(inDelim = x)
      }

      opt[String]("filter1") abbr ("f1") action { (x, options) =>
        options.copy(filter1 = x)
      } text ("String (or regex) whose presence indicates a datum for the primary item set (optional). Default: no filter, all data is used")

      opt[String]("filter2") abbr ("f2") action { (x, options) =>
        options.copy(filter2 = x)
      } text ("String (or regex) whose presence indicates a datum for the secondary item set (optional). If not present no secondary dataset is collected.")

      opt[Int]("rowIDPosition") abbr ("rc") action { (x, options) =>
        options.copy(rowIDPosition = x)
      } text ("Column number (0 based Int) containing the row ID string (optional). Default: 0") validate { x =>
        if (x >= 0) success else failure("Option --rowIDColNum must be >= 0")
      }

      opt[Int]("itemIDPosition") abbr ("ic") action { (x, options) =>
        options.copy(itemIDPosition = x)
      } text ("Column number (0 based Int) containing the item ID string (optional). Default: 1") validate { x =>
        if (x >= 0) success else failure("Option --itemIDColNum must be >= 0")
      }

      opt[Int]("filterPosition") abbr ("fc") action { (x, options) =>
        options.copy(filterPosition = x)
      } text ("Column number (0 based Int) containing the filter string (optional). Default: -1 for no filter") validate { x =>
        if (x >= -1) success else failure("Option --filterColNum must be >= -1")
      }

      note("\nUsing all defaults the input is expected of the form: \"userID<tab>itemId\" or \"userID<tab>itemID<tab>any-text...\" and all rows will be used")

      //File finding strategy--not driver specific
      note("\nFile input options:")
      opt[Unit]('r', "recursive") action { (_, options) =>
        options.copy(recursive = true)
      } text ("Searched the -i path recursively for files that match --filenamePattern (optional), Default: false")

      opt[String]("filenamePattern") abbr ("fp") action { (x, options) =>
        options.copy(filenamePattern = x)
      } text ("Regex to match in determining input files (optional). Default: filename in the --input option or \"^part-.*\" if --input is a directory")

      //Drm output schema--not driver specific, drm specific
      note("\nOutput text file schema options:")
      opt[String]("rowKeyDelim") abbr ("rd") action { (x, options) =>
        options.copy(rowKeyDelim = x)
      } text ("Separates the rowID key from the vector values list (optional). Default: \"\\t\"")

      opt[String]("columnIdStrengthDelim") abbr ("cd") action { (x, options) =>
        options.copy(columnIdStrengthDelim = x)
      } text ("Separates column IDs from their values in the vector values list (optional). Default: \":\"")

      opt[String]("tupleDelim") abbr ("td") action { (x, options) =>
        options.copy(tupleDelim = x)
      } text ("Separates vector tuple values in the values list (optional). Default: \",\"")

      //Spark config options--not driver specific
      note("\nSpark config options:")
      opt[String]("sparkExecutorMem") abbr ("sem") action { (x, options) =>
        options.copy(sparkExecutorMem = x)
      } text ("Max Java heap available as \"executor memory\" on each node (optional). Default: 4g")

      note("\nDefault delimiters will produce output of the form: \"itemID1<tab>itemID2:value2,itemID10:value10...\"")

      //Driver notes--driver specific
      note("\nNote: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.\n")

      help("help") abbr ("h") text ("prints this usage text\n")

      checkConfig { c =>
        if (c.filterPosition == c.itemIDPosition
            || c.filterPosition == c.rowIDPosition
            || c.rowIDPosition == c.itemIDPosition)
          failure("The row, item, and filter positions must be unique.") else success
      }

      //check for option consistency, probably driver specific
      checkConfig { c =>
        if (c.filter1 != null && c.filter2 != null && c.filter1 == c.filter2) failure("If" +
          " using filters they must be unique.") else success
      }

    }

    //repeated code, should this be put base MahoutDriver somehow?
    parser.parse(args, Options()) map { opts =>
      options = opts
      process
    }

  }

  override def start(masterUrl: String = options.master,
      appName: String = options.appName,
      customJars:Traversable[String] = Traversable.empty[String]):
    Unit = {

    sparkConf.set("spark.kryo.referenceTracking", "false")
      .set("spark.kryoserializer.buffer.mb", "200")
      .set("spark.executor.memory", options.sparkExecutorMem)

    super.start(masterUrl, appName)

    val readSchema1 = new Schema("delim" -> options.inDelim, "filter" -> options.filter1,
        "rowIDPosition" -> options.rowIDPosition,
        "columnIDPosition" -> options.itemIDPosition,
        "filterPosition" -> options.filterPosition)

    reader1 = new TextDelimitedIndexedDatasetReader(readSchema1)

    if (options.filterPosition != -1 && options.filter2 != null) {
      val readSchema2 = new Schema("delim" -> options.inDelim, "filter" -> options.filter2,
          "rowIDPosition" -> options.rowIDPosition,
          "columnIDPosition" -> options.itemIDPosition,
          "filterPosition" -> options.filterPosition)

      reader2 = new TextDelimitedIndexedDatasetReader(readSchema2)
    }

    writeSchema = new Schema(
        "rowKeyDelim" -> options.rowKeyDelim,
        "columnIdStrengthDelim" -> options.columnIdStrengthDelim,
        "tupleDelim" -> options.tupleDelim)

    writer = new TextDelimitedIndexedDatasetWriter(writeSchema)

  }

  private def readIndexedDatasets: Array[IndexedDataset] = {

    val inFiles = FileSysUtils(options.input, options.filenamePattern, options.recursive).uris

    if (inFiles.isEmpty) {
      Array()
    } else {

      val selfSimilarityDataset = IndexedDataset(reader1.readFrom(inFiles))

      if (options.filterPosition != -1 && options.filter2 != null) {
        // todo: needs to support more than one cross-similarity indicator
        val crossSimilarityDataset1 = IndexedDataset(reader2.readFrom(inFiles))
        Array(selfSimilarityDataset, crossSimilarityDataset1)
      } else {
        Array(selfSimilarityDataset)
      }

    }

  }

  override def process: Unit = {
    start()

    val indexedDatasets = readIndexedDatasets

    // todo: allow more than one cross-similarity matrix?
    val indicatorMatrices = {
      if (indexedDatasets.length > 1) {
        CooccurrenceAnalysis.cooccurrences(indexedDatasets(0).matrix, options.randomSeed, options.maxSimilaritiesPerItem, options.maxPrefs, Array(indexedDatasets(1).matrix))
      } else {
        CooccurrenceAnalysis.cooccurrences(indexedDatasets(0).matrix, options.randomSeed, options.maxSimilaritiesPerItem, options.maxPrefs)
      }
    }

    // self similarity
    // the next two lines write the drm using a Writer class
    // val selfIndicatorDataset = new IndexedDataset(indicatorMatrices(0), indexedDatasets(0).columnIDs, indexedDatasets(0).columnIDs)
    // writeStore.writeTo(selfIndicatorDataset, options.output + "indicator-matrix")

    // an alternative is to create a version of IndexedDataset that knows how to write itself
    val selfIndicatorDataset = new IndexedDatasetTextDelimitedWriteable(indicatorMatrices(0), indexedDatasets(0).columnIDs,
      indexedDatasets(0).columnIDs, writeSchema)
    selfIndicatorDataset.writeTo(options.output + "indicator-matrix")

    // todo: needs to support more than one cross-similarity indicator
    if (indexedDatasets.length > 1) {

      val crossIndicatorDataset = new IndexedDataset(indicatorMatrices(1), indexedDatasets(0).columnIDs, indexedDatasets(1).columnIDs) // cross similarity
      writer.writeTo(crossIndicatorDataset, options.output + "cross-indicator-matrix")

    }

    stop
  }

  // Default values go here, any "_" or null should be "required" in the Parser or flags an unused option
  // todo: support two input streams for cross-similarity, maybe assume one schema for all inputs
  case class Options(
      master: String = "local",
      sparkExecutorMem: String = "4g",
      appName: String = "ItemSimilarityJob",
      randomSeed: Int = System.currentTimeMillis().toInt,
      recursive: Boolean = false,
      input: String = null,
      output: String = null,
      filenamePattern: String = "^part-.*",
      maxSimilaritiesPerItem: Int = 100,
      maxPrefs: Int = 500,
      minPrefs: Int = 1,
      rowIDPosition: Int = 0,
      itemIDPosition: Int = 1,
      filterPosition: Int = -1,
      filter1: String = null,
      filter2: String = null,
      inDelim: String = "[,\t, ]",
      rowKeyDelim: String = "\t",
      columnIdStrengthDelim: String = ":",
      tupleDelim: String = ",")

}
