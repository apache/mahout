package org.apache.mahout.sparkbindings.shell

import org.apache.spark.repl.SparkILoop
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.mahout.sparkbindings._
import scala.tools.nsc.Properties
import scala.Some

class MahoutSparkILoop extends SparkILoop {

  private val postInitScript =
      "import org.apache.mahout.math.Vector" ::
      "import org.apache.mahout.math.scalabindings._" ::
      "import RLikeOps._" ::
      "import org.apache.mahout.sparkbindings._" ::
      "import drm._" ::
      "import RLikeDrmOps._" ::
      "org.apache.spark.storage.StorageLevel" ::
      "implicit val _sc = sc" ::
      Nil

  override protected def postInitialization() {
    super.postInitialization()
    val intp: MahoutSparkILoop = this
    intp.beQuietDuring {
      postInitScript.foreach(command(_))
    }
  }

  override def createSparkContext(): SparkContext = {
    val execUri = System.getenv("SPARK_EXECUTOR_URI")
    val master = this.master match {
      case Some(m) => m
      case None => {
        val prop = System.getenv("MASTER")
        if (prop != null) prop else "local"
      }
    }
    val jars = SparkILoop.getAddedJars.map(new java.io.File(_).getAbsolutePath)
    val conf = new SparkConf()
        .set("spark.repl.class.uri", intp.classServer.uri)

    if (execUri != null) {
      conf.set("spark.executor.uri", execUri)
    }

    sparkContext = mahoutSparkContext(
      masterUrl = master,
      appName = "Mahout Spark Shell",
      customJars = jars,
      sparkConf = conf
    )
    echo("Created spark context..")
    sparkContext
  }

  override def prompt: String = "mahout> "

  override def printWelcome(): Unit = {
    echo(
      """
                         _                 _
         _ __ ___   __ _| |__   ___  _   _| |_
        | '_ ` _ \ / _` | '_ \ / _ \| | | | __|
        | | | | | | (_| | | | | (_) | |_| | |_
        |_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 1.0

      """)
    import Properties._
    val welcomeMsg = "Using Scala %s (%s, Java %s)".format(
      versionString, javaVmName, javaVersion)
    echo(welcomeMsg)
    echo("Type in expressions to have them evaluated.")
    echo("Type :help for more information.")
  }
}
