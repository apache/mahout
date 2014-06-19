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
 * Command line interface for [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]]. It reads text lines
 * that contain (row id, column id, ...). The IDs are user specified strings which will be preserved in the
 * output. The individual tuples will be accumulated into a matrix and [[org.apache.mahout.cf.CooccurrenceAnalysis.cooccurrences( )]]
 * will be used to calculate row-wise self-similarity, or when using filters, will generate two
 * matrices and calculate both the self similarity of the primary matrix and the row-wise similarity of the primary
 * to the secondary. Returns one or two directories of text files formatted as specified in the options.
 */
object ItemSimilarityDriver extends MahoutDriver {
  //todo: Should also take two input streams and do cross similarity with no filter required.
  //todo: Required: should work on the old Mahout TDF that allows values (but ignores them), required for examples

  private var options: Options = _
  private var readStore1: TextDelimitedIndexedDatasetReader = _
  private var readStore2: TextDelimitedIndexedDatasetReader = _
  private var writeStore: TextDelimitedIndexedDatasetWriter = _
  private var writeSchema: Schema = _

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {
    val parser = new MahoutOptionParser[Options]("ItemSimilarity") {
      head("ItemSimilarity", "Spark")
      note("Input, output options")
      opt[String]('i', "input") required() action { (x, options) =>
        options.copy(input = x)
      } text ("Path for input. It may be a filename or directory name and can be a local file path or an HDFS URI (required).")
      opt[String]('o', "output") required() action { (x, options) =>
        options.copy(output = x)
      } text ("Output will be in sub-directories stored here so this must be a directory path (required).")
      note("\nAlgorithm control options:")
      opt[String]("master") abbr ("ma") text ("URL for the Spark Master. (optional). Default: 'local'") action { (x, options) =>
        options.copy(master = x)
      }
      opt[Int]("maxPrefs") abbr ("mppu") action { (x, options) =>
        options.copy(maxPrefs = x)
      } text ("Max number of preferences to consider per user or item, users or items with more preferences will"+
        " be sampled down (optional). Default: 500") validate { x =>
        if (x > 0) success else failure("Option --maxPrefs must be > 0")
      }
      opt[Int]("minPrefs") abbr ("mp") action { (x, options) =>
        options.copy(minPrefs = x)
      } text ("Ignore users with less preferences than this (optional). Default: 1") validate { x =>
        if (x > 0) success else failure("Option --minPrefs must be > 0")
      }

      opt[Int]('m', "maxSimilaritiesPerItem") action { (x, options) =>
        options.copy(maxSimilaritiesPerItem = x)
      } text ("Try to cap the number of similar items for each item to this number (optional). Default: 100") validate { x =>
        if (x > 0) success else failure("Option --maxSimilaritiesPerItem must be > 0")
      }
      opt[Int]("randomSeed") abbr ("rs") action { (x, options) =>
        options.copy(randomSeed = x)
      } text ("Int to seed random number generator (optional). Default: Uses time to generate a seed") validate { x =>
        if (x > 0) success else failure("Option --randomSeed must be > 0")
      }
      note("\nInput text file schema options:")
      opt[String]("inDelim") abbr ("d") text ("Input delimiter character (optional). Default: '\\t'") action { (x, options) =>
        options.copy(inDelim = x)
      }
      opt[String]("filter1") abbr ("f1") action { (x, options) =>
        options.copy(filter1 = x)
      } text ("String whose presence indicates a datum for the primary item set, can be a regex (optional)."+
        " Default: no filtered is applied, all is used")
      opt[String]("filter2") abbr ("f2") action { (x, options) =>
        options.copy(filter2 = x)
      } text ("String whose presence indicates a datum for the secondary item set, can be a regex (optional)."+
        " Used in cross-cooccurrence. Default: no secondary filter is applied")
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
      note("\nDefault input schema will accept: 'userID<tab>itemId' or 'userID<tab>itemID<tab>any-text...' and all rows will be used")
      note("\nFile input options:")
      opt[Unit]('r', "recursive") action { (_, options) =>
        options.copy(recursive = true)
      } text ("The input path should be searched recursively for files that match the filename pattern from -fp (optional),"+
        " Default: false")
      opt[String]("filenamePattern") abbr ("fp") action { (x, options) =>
        options.copy(filenamePattern = x)
      } text ("Regex to match in determining input files (optional). Default: filename in the --input option or '^part-.*'"+
        " if --input is a directory")
      note("\nOutput text file schema options:")
      opt[String]("outDelim1") abbr ("od1") action { (x, options) =>
        options.copy(outDelim1 = x)
      } text ("Primary output inDelim value, used to separate row IDs from the similar items list (optional). Default: '\\t'")
      opt[String]("outDelim2") abbr ("od2") action { (x, options) =>
        options.copy(outDelim2 = x)
      } text ("Secondary output inDelim value, used to separate item IDs from their values in the similar items list"+
        " (optional). Default: ':'")
      opt[String]("outDelim3") abbr ("od3") action { (x, options) =>
        options.copy(outDelim3 = x)
      } text ("Last inDelim value, used to separate (itemID:value) tuples in the similar items list. (optional). Default: ','")
      note("\nDefault delimiters will produce output of the form: 'itemID1<tab>>itemID2:value2,itemID10:value10...'")
      note("\nNote: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.\n")
      help("help") abbr ("h") text ("prints this usage text\n")
      checkConfig { c =>
        if (c.filterPosition == c.itemIDPosition || c.filterPosition == c.rowIDPosition || c.rowIDPosition == c.itemIDPosition) failure("The row, item, and filter positions must be unique.") else success
      }

      checkConfig { c =>
        if (c.filter1 != null && c.filter2 != null && c.filter1 == c.filter2) failure("If using filters they must be unique.") else success
      }

    }

    parser.parse(args, Options()) map { opts =>
      options = opts
      process
    }

  }

  private def readIndexedDatasets: Array[IndexedDataset] = {

    val inFiles = FileSysUtils(options.input, options.filenamePattern, options.recursive).uris

    if (inFiles.isEmpty) {
      Array()
    } else {

      val indexedDataset1 = IndexedDataset(readStore1.readFrom(inFiles))

      if (options.filterPosition != -1 && options.filter2 != null) {
        val indexedDataset2 = IndexedDataset(readStore2.readFrom(inFiles))
        Array(indexedDataset1, indexedDataset2)
      } else {
        Array(indexedDataset1)
      }

    }

  }

  override def start(masterUrl: String = options.master, appName: String = options.appName,
                     customJars: Traversable[String] = Traversable.empty[String]): Unit = {

    sparkConf.set("spark.kryo.referenceTracking", "false")
      .set("spark.kryoserializer.buffer.mb", "100")
      .set("spark.executor.memory", "2g")

    super.start(masterUrl, appName, customJars)
    //implicit val mc = this.mc

    val readSchema1 = new Schema("delim" -> options.inDelim, "filter" -> options.filter1,
        "rowIDPosition" -> options.rowIDPosition, "columnIDPosition" -> options.itemIDPosition,
        "filterPosition" -> options.filterPosition)
    readStore1 = new TextDelimitedIndexedDatasetReader(readSchema1)

    if (options.filterPosition != -1 && options.filter2 != null) {
      val readSchema2 = new Schema("delim" -> options.inDelim, "filter" -> options.filter2,
          "rowIDPosition" -> options.rowIDPosition, "columnIDPosition" -> options.itemIDPosition,
          "filterPosition" -> options.filterPosition)
      readStore2 = new TextDelimitedIndexedDatasetReader(readSchema2)
    }

    writeSchema = new Schema("delim1" -> options.outDelim1, "delim2" -> options.outDelim2, "delim3" -> options.outDelim3)
    writeStore = new TextDelimitedIndexedDatasetWriter(writeSchema)

  }

  override def process: Unit = {
    start()

    val indexedDatasets = readIndexedDatasets

    val indicatorMatrices = CooccurrenceAnalysis.cooccurrences(indexedDatasets(0).matrix, options.randomSeed, options.maxSimilaritiesPerItem, options.maxPrefs, Array(indexedDatasets(1).matrix))

    // self similarity
    // the next two lines write the drm using a Writer class
    // val selfIndicatorDataset = new IndexedDataset(indicatorMatrices(0), indexedDatasets(0).columnIDs, indexedDatasets(0).columnIDs)
    // writeStore.writeTo(selfIndicatorDataset, options.output + "indicator-matrix")

    // an alternative is to create a version of IndexedDataset that knows how to write itself
    val selfIndicatorDataset = new IndexedDatasetTextDelimitedWriteable(indicatorMatrices(0), indexedDatasets(0).columnIDs,
      indexedDatasets(0).columnIDs, writeSchema)
    selfIndicatorDataset.writeTo(options.output + "indicator-matrix")

    if (indexedDatasets.length > 1) {

      val crossIndicatorDataset = new IndexedDataset(indicatorMatrices(1), indexedDatasets(0).columnIDs, indexedDatasets(1).columnIDs) // cross similarity
      writeStore.writeTo(crossIndicatorDataset, options.output + "cross-indicator-matrix")

    }

    stop
  }

  // Default values go here, any '_' or null should be 'required' in the Parser or flags an unused option
  // todo: support two input streams for cross-similarity, maybe assume one schema for both inputs
  case class Options(
      master: String = "local",
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
      inDelim: String = ",",
      outDelim1: String = "\t",
      outDelim2: String = ":",
      outDelim3: String = ",")

}
