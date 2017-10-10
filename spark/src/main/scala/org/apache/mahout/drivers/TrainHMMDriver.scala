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

import org.apache.mahout.common.Hadoop2HDFSUtil
import org.apache.mahout.math.drm
import org.apache.mahout.math.drm.DrmLike

import scala.collection.immutable.HashMap


object TrainHMMDriver extends MahoutSparkDriver {
  // define only the options specific to TrainHMM
  private final val trainHMMOptipns = HashMap[String, Any](
    "appName" -> "TrainHMMDriver")

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-trainhmm") {
      head("spark-trainhmm", "Mahout 0.10.0")

      // Input output options, non-driver specific
      parseIOOptions(numInputs = 1)

      // Algorithm control options--driver specific
      opts = opts ++ trainHMMOptipns
      note("\nAlgorithm control options:")

      // Overwrite the output directory (with the model) if it exists?  Default: false
      opts = opts + ("numberOfHiddenStates" -> false)
      opt[Int]("numberOfHiddenStates") abbr "nh" action { (_, options) =>
        options + ("numberOfHiddenStates" -> true)
      } text "Number of hidden states"

      opts = opts + ("numberOfObservableSymbols" -> false)
      opt[Int]("numberOfOfObservableSymbols") abbr "no" action { (_, options) =>
        options + ("numberOfObservableSymbols" -> true)
      } text "Number of observable symbols"

      // epsilon
      opts = opts + ("epsilon" -> 1.0)
      opt[Double]("epsilon") abbr "e" action { (x, options) =>
        options + ("epsilon" -> x)
      } text "Convergence threshold" validate { x =>
        if (x > 0) success else failure("Option --epsilon must be > 0")
      }

      opts = opts + ("maxNumberOfIterations" -> false)
      opt[Int]("maxNumberOfIterations") abbr "no" action { (_, options) =>
        options + ("maxNumberOfIterations" -> true)
      } text "Maximum Number of Iterations"
      
      // default scale is false
      opts = opts + ("scale" -> false)
      opt[Unit]("scale") abbr "s" action { (_, options) =>
        options + ("scale" -> true)
      } text "Rescale forward and backward variables after each iteration, Default: false."

      // Spark config options--not driver specific
      parseSparkOptions()

      help("help") abbr "h" text "prints this usage text\n"

    }
    parser.parse(args, parser.opts) map { opts =>
      parser.opts = opts
      process
    }
  }

  override def process(): Unit = {
    start()

    stop()
  }

}
