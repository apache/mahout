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

import org.apache.spark.SparkConf

/** Adds parsing of Spark specific options to the option parser */
class MahoutSparkOptionParser(programName: String) extends MahoutOptionParser(programName: String){

  def parseSparkOptions()(implicit sparkConf: SparkConf) = {
    opts = opts ++ MahoutOptionParser.SparkOptions
    opts = opts + ("appName" -> programName)
    note("\nSpark config options:")

    opt[String]("master") abbr "ma" text ("Spark Master URL (optional). Default: \"local\". Note that you can " +
      "specify the number of cores to get a performance improvement, for example \"local[4]\"") action { (x, options) =>
        options + ("master" -> x)
    }

    opt[String]("sparkExecutorMem") abbr "sem" text ("Max Java heap available as \"executor memory\" on each " +
      "node (optional). Default: as Spark config specifies") action { (x, options) =>
        options + ("sparkExecutorMem" -> x)
    }

    opt[(String, String)]("define") abbr "D" unbounded() foreach { case (k, v) =>
      sparkConf.set(k, v)
    } validate { x =>
      if (x._2 != "") success else failure("Value <sparkConfValue> must be non-blank")
    } keyValueName("<sparkConfKey>", "<sparkConfValue>") text ("Set the <sparkConfKey> to <sparkConfValue> before " +
      "creating this job's Spark context (optional)")

  }
}
