package org.apache.mahout.sparkbindings

import java.io.{Closeable, File}

import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.util.IOUtilsScala
import org.scalatest.FunSuite

import scala.collection._

/**
 * @author dmitriy
 */
class SparkBindingsSuite extends FunSuite with DistributedSparkSuite {

  // This test will succeed only when MAHOUT_HOME is set in the environment. So we keep it for
  // diagnostic purposes around, but we probably don't want it to run in the Jenkins, so we'd
  // let it to be ignored.
  ignore("context jars") {
    System.setProperty("mahout.home", new File("..").getAbsolutePath/*"/home/dmitriy/projects/github/mahout-commits"*/)
    val closeables = new mutable.ListBuffer[Closeable]()
    try {
      val mahoutJars = findMahoutContextJars(closeables)
      mahoutJars.foreach {
        println(_)
      }

      mahoutJars.size should be > 0
      mahoutJars.size shouldBe 4
    } finally {
      IOUtilsScala.close(closeables)
    }

  }

}
