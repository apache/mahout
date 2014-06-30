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

/** Extend this class to create a Mahout CLI driver. Minimally you must override process and main.
  * Also define a command line parser and default options or fill in the following template:
  * {{{
  *   object SomeDriver extends MahoutDriver {
  *     override def main(args: Array[String]): Unit = {
  *       val parser = new MahoutOptionParser[Options]("Job Name") {
  *         head("Job Name", "Spark")
  *         note("Various CLI options")
  *         //see https://github.com/scopt/scopt for a good Scala option parser, which MahoutOptionParser extends
  *       }
  *       parser.parse(args, Options()) map { opts =>
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
  *
  *     //Default values go here, any '_' or null should be 'required' in the Parser or flags an unused option
  *     case class Options(
  *       appName: String = "Job Name", ...
  *     )
  *   }
  * }}}
  */
abstract class MahoutDriver {
  implicit var mc: DistributedContext = _
  implicit val sparkConf = new SparkConf()

  /** Creates a Spark context to run the job inside.
    * Creates a Spark context to run the job inside. Override to set the SparkConf values specific to the job,
    * these must be set before the context is created.
    * @param masterUrl Spark master URL
    * @param appName  Name to display in Spark UI
    * @param customJars List of paths to custom jars
    * */
  protected def start(masterUrl: String, appName: String,
            customJars:Traversable[String] = Traversable.empty[String]) : Unit = {
    mc = mahoutSparkContext(masterUrl, appName, customJars, sparkConf)
  }

  /** Override (optionally) for special cleanup */
  protected def stop: Unit = {
    mc.close
  }

  /** This is wher you do the work, call start first, then before exiting call stop */
  protected def process: Unit

  /** Parse command line and call process */
  def main(args: Array[String]): Unit
}
