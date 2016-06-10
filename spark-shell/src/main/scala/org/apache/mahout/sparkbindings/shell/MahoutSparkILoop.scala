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


class MahoutSparkILoop extends SparkILoop {

  private var _interp: SparkILoop = _

  private var sdc: SparkDistributedContext = _

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

  def getSparkDistributedContext: SparkDistributedContext  = sdc

  // Hack: for some very unclear reason, log4j is not picking up log4j.properties in Spark conf/ even
  // though the latter is added to the classpath. So we force it to pick it.
  PropertyConfigurator.configure(getMahoutHome() + "/conf/log4j.properties")

  System.setProperty("scala.usejavacp", "true")

  _interp = this

  // It looks like we need to initialize this too, since some Spark shell initilaization code
  // expects it
  org.apache.spark.repl.Main.interp = _interp

  _interp.setPrompt("mahout> ")

  // sparkILoop.echo(...) is private so we create our own here.
  def echoToShell(str: String): Unit = {
    _interp.out.println(str)
  }

  // create a spark context as a mahout SparkDistributedContext.
  // store the SparkDistributedContext for decleration in the intreperer session.
  override def createSparkContext(): SparkContext = {
    val execUri = System.getenv("SPARK_EXECUTOR_URI")
    val master = _interp.master match {
      case Some(m) => m
      case None => {
        val prop = System.getenv("MASTER")
        if (prop != null) prop else "local"
      }
    }

    val jars = SparkILoop.getAddedJars.map(new java.io.File(_).getAbsolutePath)
    val conf = new SparkConf().set("spark.repl.class.uri", _interp.classServerUri)

    if (execUri != null) {
      conf.set("spark.executor.uri", execUri)
    }

    // set default value of spark.executor.memory to 1g
    if(!conf.contains("spark.executor.memory")) {
      conf.set("spark.executor.memory", "1g")
    }

    // set default if not already set- this is useful in local mode
    if(!conf.contains("spark.kryoserializer.buffer.max")) {
      conf.set("spark.kryoserializer.buffer.max", "1g")
    }

    sdc = mahoutSparkContext(
      masterUrl = master,
      appName = "Mahout Spark Shell",
      customJars = jars,
      sparkConf = conf
    )

    _interp.sparkContext = sdc

    echoToShell("Created spark context..")
    sparkContext
  }

  // this is technically not part of Spark's explicitly defined Developer API though
  // nothing in the SparkILoopInit.scala file is marked as such.
  override def initializeSpark() {

    _interp.beQuietDuring {

      // get the spark context, at the same time create and store a mahout distributed context.
      _interp.interpret("""
         @transient val sc = {
           val _sc = org.apache.spark.repl.Main.interp.createSparkContext()
           _sc
         }
                        """)
      echoToShell("Spark context is available as \"val sc\".")

      // retrieve the stored mahout SparkDistributedContext.
      _interp.interpret("""
         @transient implicit val sdc: org.apache.mahout.sparkbindings.SparkDistributedContext =
            org.apache.spark.repl.Main.interp
             .asInstanceOf[org.apache.mahout.sparkbindings.shell.MahoutSparkILoop]
             .getSparkDistributedContext
                        """)
      echoToShell("Mahout distributed context is available as \"implicit val sdc\".")

      // create a SQL Context.
      _interp.interpret("""
         @transient val sqlContext = {
           val _sqlContext = org.apache.spark.repl.Main.interp.createSQLContext()
           _sqlContext
         }
                        """)
      _interp.interpret("import org.apache.spark.SparkContext._")
      _interp.interpret("import sqlContext.implicits._")
      _interp.interpret("import sqlContext.sql")
      _interp.interpret("import org.apache.spark.sql.functions._")
      echoToShell("SQL context available as \"val sqlContext\".")

    }
  }

  // this is technically not part of Spark's explicitly defined Developer API though
  // nothing in the SparkILoopInit.scala file is marked as such.
  override protected def postInitialization() {
    super.postInitialization()
    _interp.beQuietDuring {
      postInitImports.foreach(_interp.interpret)
    }
  }

  // this is technically not part of Spark's explicitly defined Developer API though
  // nothing in the SparkILoopInit.scala file is marked as such..
  override def printWelcome(): Unit = {
    echoToShell(
      """
                         _                 _
         _ __ ___   __ _| |__   ___  _   _| |_
        | '_ ` _ \ / _` | '_ \ / _ \| | | | __|
        | | | | | | (_| | | | | (_) | |_| | |_
        |_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 0.12.2

      """)
    import Properties._
    val welcomeMsg = "Using Scala %s (%s, Java %s)".format(
      versionString, javaVmName, javaVersion)
    echoToShell(welcomeMsg)
    echoToShell("Type in expressions to have them evaluated.")
    echoToShell("Type :help for more information.")
  }

}

