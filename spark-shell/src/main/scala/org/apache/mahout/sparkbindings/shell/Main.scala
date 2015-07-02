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

package org.apache.mahout.sparkbindings.shell

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.repl.SparkILoop
import scala.tools.nsc.Properties
import org.apache.mahout.sparkbindings._


class MahoutLoop extends SparkILoop{
  private var _interp: SparkILoop = _

  private val postInitImports =
    "import org.apache.mahout.math._" ::
      "import scalabindings._" ::
      "import RLikeOps._" ::
      "import drm._" ::
      "import RLikeDrmOps._" ::
      "import decompositions._" ::
      "import org.apache.mahout.sparkbindings._" ::
      "import collection.JavaConversions._" ::
      Nil




    // Hack: for some very unclear reason, log4j is not picking up log4j.properties in Spark conf/ even
    // though the latter is added to the classpath. So we force it to pick it.
    PropertyConfigurator.configure(getMahoutHome() + "/conf/log4j.properties")

    System.setProperty("scala.usejavacp", "true")
    _interp = this
    // It looks like we need to initialize this too, since some Spark shell initilaization code
    // expects it
    org.apache.spark.repl.Main.interp = _interp


    _interp.setPrompt("mahout> ")


   override def createSparkContext(): SparkContext = {
      val execUri = System.getenv("SPARK_EXECUTOR_URI")
      val master = _interp.master match {
        case Some(m) => m
        case None => {
          val prop = System.getenv("MASTER")
          if (prop != null) prop else "local"
        }
      }

//      val iMain: SparkIMain = SparkILoop.loopToInterpreter(_interp)
//     print("\n Imain: "+iMain.getClass.getName)
//     //System.exit(0)
//
//     val classServerUri_ : String = iMain.classServerUri.toString
//     System.out.println("!!!!!!!"+classServerUri_)

      val jars = SparkILoop.getAddedJars.map(new java.io.File(_).getAbsolutePath)
      val conf = new SparkConf().set("spark.repl.class.uri", _interp.classServerUri)

      if (execUri != null) {
        conf.set("spark.executor.uri", execUri)
      }

      conf.set("spark.executor.memory", "1g")

      _interp.sparkContext= mahoutSparkContext(
          masterUrl = master,
          appName = "Mahout Spark Shell",
          customJars = jars,
          sparkConf = conf
        )

      echoToShell("Created spark context..")
      sparkContext
  }

  override def initializeSpark() {
      _interp.beQuietDuring {
        _interp.interpret("""

         @transient implicit val sdc: org.apache.mahout.math.drm.DistributedContext =
            new org.apache.mahout.sparkbindings.SparkDistributedContext(
            org.apache.spark.repl.Main.interp.createSparkContext())

                """)
        _interp.interpret("import org.apache.spark.SparkContext._")
        echoToShell("Mahout distributed context is available as \"implicit val sdc\".")
      }
    }

    def sparkCleanUp() {
      echoToShell("Stopping Spark context.")
      _interp.beQuietDuring {
        _interp.interpret("sdc.stop()")
      }
    }

  override protected def postInitialization() {
    super.postInitialization()
    //val intp: MahoutSparkILoop = this
    _interp.beQuietDuring {
      postInitImports.foreach(_interp.interpret(_))
    }
  }

  def echoToShell(str: String): Unit = {
    _interp.out.println(str)
  }

  override def printWelcome(): Unit = {
    echoToShell(
      """
                         _                 _
         _ __ ___   __ _| |__   ___  _   _| |_
        | '_ ` _ \ / _` | '_ \ / _ \| | | | __|
        | | | | | | (_| | | | | (_) | |_| | |_
        |_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 0.11.0
      """)
    import Properties._
    val welcomeMsg = "Using Scala %s (%s, Java %s)".format(
      versionString, javaVmName, javaVersion)
    echoToShell(welcomeMsg)
    echoToShell("Type in expressions to have them evaluated.")
    echoToShell("Type :help for more information.")
  }

}

object Main {
  private var _interp: SparkILoop = _

  def main(args: Array[String]) {
    PropertyConfigurator.configure(getMahoutHome() + "/conf/log4j.properties")

    System.setProperty("scala.usejavacp", "true")
    _interp = new MahoutLoop()
    // It looks like we need to initialize this too, since some Spark shell initilaization code
    // expects it
    org.apache.spark.repl.Main.interp = _interp
    _interp.process(args)

  }
}




//log.info("Mahout spark shell waking up.")

//private val postInitScript =
//"import org.apache.mahout.math._" ::
//"import scalabindings._" ::
//"import RLikeOps._" ::
//"import drm._" ::
//"import RLikeDrmOps._" ::
//"import decompositions._" ::
//"import org.apache.mahout.sparkbindings._" ::
//"import collection.JavaConversions._" ::
//Nil
//
//override protected def postInitialization() {
//super.postInitialization()
//val intp: MahoutSparkILoop = this
//intp.beQuietDuring {
//postInitScript.foreach(command(_))
//}
//}
//
//override def createSparkContext(): SparkContext = {
//val execUri = System.getenv("SPARK_EXECUTOR_URI")
//val master = this.master match {
//case Some(m) => m
//case None => {
//val prop = System.getenv("MASTER")
//if (prop != null) prop else "local"
//}
//}
//val jars = SparkILoop.getAddedJars.map(new java.io.File(_).getAbsolutePath)
//val conf = new SparkConf()
//.set("spark.repl.class.uri", intp.classServer.uri)
//
//if (execUri != null) {
//conf.set("spark.executor.uri", execUri)
//}
//
//conf.set("spark.executor.memory", "1g")
//
//sparkContext = mahoutSparkContext(
//masterUrl = master,
//appName = "Mahout Spark Shell",
//customJars = jars,
//sparkConf = conf
//)
//
//echo("Created spark context..")
//sparkContext
//}
//
//override def initializeSpark() {
//intp.beQuietDuring {
//command("""
//
//         @transient implicit val sdc: org.apache.mahout.math.drm.DistributedContext =
//            new org.apache.mahout.sparkbindings.SparkDistributedContext(
//            org.apache.spark.repl.Main.interp.createSparkContext())
//
//              """)
//command("import org.apache.spark.SparkContext._")
//echo("Mahout distributed context is available as \"implicit val sdc\".")
//}
//}
//
//override def sparkCleanUp() {
//echo("Stopping Spark context.")
//intp.beQuietDuring {
//command("sdc.stop()")
//}
//}
//
//override def prompt: String = "mahout> "
//
//override def printWelcome(): Unit = {
//echo(
//"""
//                   _                 _
//   _ __ ___   __ _| |__   ___  _   _| |_
//  | '_ ` _ \ / _` | '_ \ / _ \| | | | __|
//  | | | | | | (_| | | | | (_) | |_| | |_
//  |_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 0.10.0
//
//""")
//import Properties._
//val welcomeMsg = "Using Scala %s (%s, Java %s)".format(
//versionString, javaVmName, javaVersion)
//echo(welcomeMsg)
//echo("Type in expressions to have them evaluated.")
//echo("Type :help for more information.")
//}
