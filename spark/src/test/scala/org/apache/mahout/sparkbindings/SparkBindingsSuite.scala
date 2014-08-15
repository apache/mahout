package org.apache.mahout.sparkbindings

import org.scalatest.FunSuite
import java.util
import java.io.{File, Closeable}
import org.apache.mahout.common.IOUtils
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite

/**
 * @author dmitriy
 */
class SparkBindingsSuite extends FunSuite with DistributedSparkSuite {

  // This test will succeed only when MAHOUT_HOME is set in the environment. So we keep it for
  // diagnorstic purposes around, but we probably don't want it to run in the Jenkins, so we'd
  // let it to be ignored.
  ignore("context jars") {
    System.setProperty("mahout.home", new File("..").getAbsolutePath/*"/home/dmitriy/projects/github/mahout-commits"*/)
    val closeables = new util.ArrayDeque[Closeable]()
    try {
      val mahoutJars = findMahoutContextJars(closeables)
      mahoutJars.foreach {
        println(_)
      }

      mahoutJars.size should be > 0
      mahoutJars.size shouldBe 4
    } finally {
      IOUtils.close(closeables)
    }

  }

}
