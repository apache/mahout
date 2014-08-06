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

import scopt.OptionParser
import scala.collection.immutable

/** Companion object defines default option groups for reference in any driver that needs them */
object MahoutOptionParser {
  // set up the various default option groups
  final val GenericOptions = immutable.HashMap[String, Any](
    "randomSeed" -> System.currentTimeMillis().toInt,
    "dontAddMahoutJars" -> false,
    "writeAllDatasets" -> false)

  final val SparkOptions = immutable.HashMap[String, Any](
    "master" -> "local",
    "sparkExecutorMem" -> "2g",
    "appName" -> "Generic Spark App, Change this.")

  final val FileIOOptions = immutable.HashMap[String, Any](
    "recursive" -> false,
    "input" -> null.asInstanceOf[String],
    "input2" -> null.asInstanceOf[String],
    "output" -> null.asInstanceOf[String],
    "filenamePattern" -> "^part-.*")

  final val TextDelimitedTuplesOptions = immutable.HashMap[String, Any](
    "rowIDPosition" -> 0,
    "itemIDPosition" -> 1,
    "filterPosition" -> -1,
    "filter1" -> null.asInstanceOf[String],
    "filter2" -> null.asInstanceOf[String],
    "inDelim" -> "[,\t ]")

  final val TextDelimitedDRMOptions = immutable.HashMap[String, Any](
    "rowKeyDelim" -> "\t",
    "columnIdStrengthDelim" -> ":",
    "tupleDelim" -> " ",
    "omitStrength" -> false)
}
/** Defines oft-repeated options and their parsing. Provides the option groups and parsing helper methods to
  * keep both standarized.
  * @param programName Name displayed in help message, the name by which the driver is invoked.
  * */
class MahoutOptionParser(programName: String) extends OptionParser[Map[String, Any]](programName: String) {
  override def showUsageOnError = true

  def parseIOOptions = {
    note("Input, output options")
    opt[String]('i', "input") required() action { (x, options) =>
      options + ("input" -> x)
    } text ("Input path, may be a filename, directory name, or comma delimited list of HDFS supported URIs (required)")

    opt[String]("input2") abbr ("i2")  action { (x, options) =>
      options + ("input2" -> x)
    } text ("Secondary input path for cross-similarity calculation, same restrictions as \"--input\" (optional). Default: empty.")

    opt[String]('o', "output") required() action { (x, options) =>
      // todo: check to see if HDFS allows MS-Windows backslashes locally?
      if (x.endsWith("/")) {
        options + ("output" -> x)
      } else {
        options + ("output" -> (x + "/"))
      }
    } text ("Path for output, any local or HDFS supported URI (required)")

  }

  def parseSparkOptions = {
    note("\nSpark config options:")

    opt[String]("master") abbr ("ma") text ("Spark Master URL (optional). Default: \"local\". Note that you can specify the number of cores to get a performance improvement, for example \"local[4]\"") action { (x, options) =>
      options + ("master" -> x)
    }

    opt[String]("sparkExecutorMem") abbr ("sem") text ("Max Java heap available as \"executor memory\" on each node (optional). Default: 4g") action { (x, options) =>
      options + ("sparkExecutorMem" -> x)
    }

  }

  def parseGenericOptions = {
    note("\nGeneral config options:")
    opt[Int]("randomSeed") abbr ("rs") action { (x, options) =>
      options + ("randomSeed" -> x)
    } validate { x =>
      if (x > 0) success else failure("Option --randomSeed must be > 0")
    }

    opt[Unit]("dontAddMahoutJars") hidden() action { (_, options) =>
      options + ("dontAddMahoutJars" -> true)
    }//Hidden option, used when executing tests or calling from other code where classes are all loaded explicitly

    //output both input DRMs
    opt[Unit]("writeAllDatasets") hidden() action { (_, options) =>
      options + ("writeAllDatasets" -> true)
    }//Hidden option, though a user might want this.
  }

  def parseInputSchemaOptions{
    //Input text file schema--not driver specific but input data specific, tuples input,
    // not drms
    note("\nInput text file schema options:")
    opt[String]("inDelim") abbr ("id") text ("Input delimiter character (optional). Default: \"[,\\t]\"") action { (x, options) =>
      options + ("inDelim" -> x)
    }

    opt[String]("filter1") abbr ("f1") action { (x, options) =>
      options + ("filter1" -> x)
    } text ("String (or regex) whose presence indicates a datum for the primary item set (optional). Default: no filter, all data is used")

    opt[String]("filter2") abbr ("f2") action { (x, options) =>
      options + ("filter2" -> x)
    } text ("String (or regex) whose presence indicates a datum for the secondary item set (optional). If not present no secondary dataset is collected")

    opt[Int]("rowIDPosition") abbr ("rc") action { (x, options) =>
      options + ("rowIDPosition" -> x)
    } text ("Column number (0 based Int) containing the row ID string (optional). Default: 0") validate { x =>
      if (x >= 0) success else failure("Option --rowIDColNum must be >= 0")
    }

    opt[Int]("itemIDPosition") abbr ("ic") action { (x, options) =>
      options + ("itemIDPosition" -> x)
    } text ("Column number (0 based Int) containing the item ID string (optional). Default: 1") validate { x =>
      if (x >= 0) success else failure("Option --itemIDColNum must be >= 0")
    }

    opt[Int]("filterPosition") abbr ("fc") action { (x, options) =>
      options + ("filterPosition" -> x)
    } text ("Column number (0 based Int) containing the filter string (optional). Default: -1 for no filter") validate { x =>
      if (x >= -1) success else failure("Option --filterColNum must be >= -1")
    }

    note("\nUsing all defaults the input is expected of the form: \"userID<tab>itemId\" or \"userID<tab>itemID<tab>any-text...\" and all rows will be used")

    checkConfig { options: Map[String, Any] =>
      if (options("filterPosition").asInstanceOf[Int] == options("itemIDPosition").asInstanceOf[Int]
        || options("filterPosition").asInstanceOf[Int] == options("rowIDPosition").asInstanceOf[Int]
        || options("rowIDPosition").asInstanceOf[Int] == options("itemIDPosition").asInstanceOf[Int])
        failure("The row, item, and filter positions must be unique.") else success
    }

    //check for option consistency, probably driver specific
    checkConfig { options: Map[String, Any] =>
      if (options("filter1").asInstanceOf[String] != null.asInstanceOf[String]
        && options("filter2").asInstanceOf[String] != null.asInstanceOf[String]
        && options("filter1").asInstanceOf[String] == options("filter2").asInstanceOf[String])
        failure ("If using filters they must be unique.") else success
    }

  }

  def parseFileDiscoveryOptions = {
    //File finding strategy--not driver specific
    note("\nFile discovery options:")
    opt[Unit]('r', "recursive") action { (_, options) =>
      options + ("recursive" -> true)
    } text ("Searched the -i path recursively for files that match --filenamePattern (optional), Default: false")

    opt[String]("filenamePattern") abbr ("fp") action { (x, options) =>
      options + ("filenamePattern" -> x)
    } text ("Regex to match in determining input files (optional). Default: filename in the --input option or \"^part-.*\" if --input is a directory")

  }

  def parseDrmFormatOptions = {
    note("\nOutput text file schema options:")
    opt[String]("rowKeyDelim") abbr ("rd") action { (x, options) =>
      options + ("rowKeyDelim" -> x)
    } text ("Separates the rowID key from the vector values list (optional). Default: \"\\t\"")

    opt[String]("columnIdStrengthDelim") abbr ("cd") action { (x, options) =>
      options + ("columnIdStrengthDelim" -> x)
    } text ("Separates column IDs from their values in the vector values list (optional). Default: \":\"")

    opt[String]("tupleDelim") abbr ("td") action { (x, options) =>
      options + ("tupleDelim" -> x)
    } text ("Separates vector tuple values in the values list (optional). Default: \" \"")

    opt[Unit]("omitStrength") abbr ("os") action { (_, options) =>
      options + ("omitStrength" -> true)
    } text ("Do not write the strength to the output files (optional), Default: false.")
    note("This option is used to output indexable data for creating a search engine recommender.")

    note("\nDefault delimiters will produce output of the form: \"itemID1<tab>itemID2:value2<space>itemID10:value10...\"")
  }

}


