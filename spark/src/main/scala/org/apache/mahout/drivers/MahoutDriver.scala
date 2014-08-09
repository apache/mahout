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

import org.apache.mahout.math.drm.DistributedContext
import org.apache.spark.SparkConf
import org.apache.mahout.sparkbindings._

import scala.collection.immutable

/** Extend this class to create a Mahout CLI driver. Minimally you must override process and main.
  * Also define a Map of options for the command line parser. The following template may help:
  * {{{
  *   object SomeDriver extends MahoutDriver {
  *     // build options from some stardard CLI param groups
  *     // Note: always put the driver specific options at the last so the can override and previous options!
  *     private var options = GenericOptions ++ SparkOptions ++ FileIOOptions ++ TextDelimitedTuplesOptions ++
  *       TextDelimitedDRMOptions ++ ItemSimilarityOptions
  *
  *     override def main(args: Array[String]): Unit = {
  *       val parser = new MahoutOptionParser(programName = "spark-itemsimilarity") {
  *         head("spark-itemsimilarity", "Mahout 1.0-SNAPSHOT")
  *
  *         //Several standard option groups are usually non-driver specific so use the MahoutOptionParser methods
  *         parseGenericOptions
  *         ...
  *       }
  *       parser.parse(args, options) map { opts =>
  *         options = opts
  *         process
  *       }
  *     }
  *
  *     override def process: Unit = {
  *       start()
  *       //don't just stand there do something
  *       stop
  *     }
  *   }
  * }}}
  */
abstract class MahoutDriver {


  implicit var mc: DistributedContext = _
  implicit var sparkConf = new SparkConf()
  var _useExistingContext: Boolean = false

  /** Creates a Spark context to run the job inside.
    * Creates a Spark context to run the job inside. Override to set the SparkConf values specific to the job,
    * these must be set before the context is created.
    * @param masterUrl Spark master URL
    * @param appName  Name to display in Spark UI
    * */
  protected def start(masterUrl: String, appName: String) : Unit = {
    if (!_useExistingContext) {
      mc = mahoutSparkContext(masterUrl, appName, sparkConf = sparkConf)
    }
  }

  /** Override (optionally) for special cleanup */
  protected def stop: Unit = {
    if (!_useExistingContext) mc.close
  }

  /** This is where you do the work, call start first, then before exiting call stop */
  protected def process: Unit

  /** Parse command line and call process */
  def main(args: Array[String]): Unit

  def useContext(context: DistributedContext): Unit = {
    _useExistingContext = true
    mc = context
    sparkConf = mc.getConf
  }
}
