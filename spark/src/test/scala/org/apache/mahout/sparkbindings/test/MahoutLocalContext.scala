package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.mahout.sparkbindings._
import org.apache.mahout.test.MahoutSuite

trait MahoutLocalContext extends MahoutSuite with LoggerConfiguration {
  this: Suite =>

  protected implicit var mahoutCtx: SparkContext = _

  override protected def beforeEach() {
    super.beforeEach()

    mahoutCtx = mahoutSparkContext(masterUrl = "local[3]",
      appName = "MahoutLocalContext",
      // Do not run MAHOUT_HOME jars in unit tests.
      addMahoutJars = false,
      sparkConf = new SparkConf()
          .set("spark.kryoserializer.buffer.mb", "15")
          .set("spark.akka.frameSize", "30")
    )
  }

  override protected def afterEach() {
    if (mahoutCtx != null) {
      try {
        mahoutCtx.stop()
      } finally {
        mahoutCtx = null
      }
    }
    super.afterEach()
  }
}
