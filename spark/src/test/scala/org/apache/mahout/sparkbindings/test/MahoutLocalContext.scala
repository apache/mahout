package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.spark.SparkConf
import org.apache.mahout.sparkbindings._
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.math.drm.DistributedContext

trait MahoutLocalContext extends MahoutSuite with LoggerConfiguration {
  this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _

  override protected def beforeEach() {
    super.beforeEach()

    mahoutCtx = mahoutSparkContext(masterUrl = "local[2]",
      appName = "MahoutLocalContext",
      // Do not run MAHOUT_HOME jars in unit tests.
      addMahoutJars = false,
      sparkConf = new SparkConf()
          .set("spark.kryoserializer.buffer.mb", "15")
          .set("spark.akka.frameSize", "30")
          .set("spark.default.parallelism", "10")
    )
  }

  override protected def afterEach() {
    if (mahoutCtx != null) {
      try {
        mahoutCtx.close()
      } finally {
        mahoutCtx = null
      }
    }
    super.afterEach()
  }
}
