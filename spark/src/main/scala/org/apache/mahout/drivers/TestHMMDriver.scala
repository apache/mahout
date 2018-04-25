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

import org.apache.mahout.classifier.sequencelearning.hmm.{SparkHiddenMarkovModel, _}
import org.apache.mahout.common.Hadoop2HDFSUtil
import org.apache.mahout.math.drm
import org.apache.mahout.math._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.{CheckpointedDrm, drmParallelizeEmpty}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.{`::`, dense}
import scala.collection.immutable.HashMap
import scala.io.Source
import org.apache.mahout.sparkbindings._

object TestHMMDriver extends MahoutSparkDriver {
  // define only the options specific to TestHMM
  private final val testHMMOptions = HashMap[String, Any](
    "appName" -> "TestHMMDriver")

  /**
   * @param args  Command line args, if empty a help message is printed.
   */
  override def main(args: Array[String]): Unit = {

    parser = new MahoutSparkOptionParser(programName = "spark-testhmm") {
      head("spark-testhmm", "Mahout 0.10.0")

      // Input output options, non-driver specific
      note("Input, option")
      opt[String]('i', "input") required() action { (x, options) =>
        options + ("input" -> x)
      } text ("Input: path to trained model " +
        " (required)")

      // Algorithm control options--driver specific
      opts = opts ++ testHMMOptions
      note("\nAlgorithm control options:")

      cmd("decode").action( (_, c) => c + ("opCode" -> "decode")).
        text("decode the hidden state sequence given a observation sequence.").
        children(
          opt[String]("observation-sequence").abbr("seq").action( (x, c) =>
            c + ("seq" -> x) ).text("observed sequence"),
          opt[Boolean]("scale").abbr("sc").action( (x, c) =>
            c + ("scale" -> x) ).text("scale")
        )

      cmd("generate").action( (_, c) => c + ("opCode" -> "generate")).
        text("generate a random observation sequence of given length given a model.").
        children(
          opt[String]("sequence-length").abbr("len").action( (x, c) =>
            c + ("length" -> x) ).text("sequence length")
        )

      cmd("likelihood").action( (_, c) => c + ("opCode" -> "likelihood")).
        text("compute the likelihood of a given sequence given a model.").
        children(
          opt[String]("observation-sequence").abbr("seq").action( (x, c) =>
            c + ("seq" -> x) ).text("observed sequence"),
          opt[Boolean]("scale").abbr("sc").action( (x, c) =>
            c + ("scale" -> x) ).text("scale")
        )
      
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

    val inputPath = parser.opts("input").asInstanceOf[String]
    val inputModel = HMMModel.dfsRead(inputPath)
    val op = parser.opts("opCode").asInstanceOf[String]
    if (op == "likelihood") {
      val seqFile = parser.opts("seq").asInstanceOf[String]
      val scale:Boolean = parser.opts("scale").asInstanceOf[Boolean]
      val bufferedSource = Source.fromFile(seqFile)
      val r = bufferedSource.getLines.map(line => line.split(" ") ).map(numbers => new DenseVector(numbers.map(_.toDouble)))
      for (s <- r ) {
        val seqVec:DenseVector = s
        val likelihood:Double = SparkHiddenMarkovModel.likelihood(inputModel, seqVec, scale)
        println(s"Sequence likelihood: $likelihood")
      }

      bufferedSource.close
    } else if (op == "generate") {
      val len:Int = parser.opts("seq").asInstanceOf[Int]
      val (observationSeq, hiddenSeq) = SparkHiddenMarkovModel.generate(inputModel, len, System.currentTimeMillis().toInt)
    } else if (op == "decode") {
      val seqFile = parser.opts("seq").asInstanceOf[String]
      val scale:Boolean = parser.opts("scale").asInstanceOf[Boolean]
      val bufferedSource = Source.fromFile(seqFile)
      val r = bufferedSource.getLines.map(line => line.split(" ") ).map(numbers => new DenseVector(numbers.map(_.toDouble)))
      for (s <- r ) {
        val seqVec:DenseVector = s
        val hiddenSeq:DenseVector = SparkHiddenMarkovModel.decode(inputModel, seqVec, scale)
        println(s"hidden sequence: $hiddenSeq")
      }

      bufferedSource.close
    } else {
      println("Unknown command.")
    }

    stop()
  }

}
