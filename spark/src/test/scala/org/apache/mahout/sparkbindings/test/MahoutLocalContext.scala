package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.spark.SparkContext
import org.apache.mahout.sparkbindings._
import org.apache.mahout.test.MahoutSuite

/**
 * @author dmitriy
 */
trait MahoutLocalContext extends MahoutSuite with LoggerConfiguration {
  this: Suite =>

  protected implicit var mahoutCtx: SparkContext = _

  // Additional jars?
  protected val buildJars = Traversable.empty[String]

  override protected def beforeEach() {
    super.beforeEach()
    mahoutCtx = mahoutSparkContext(masterUrl = "local", appName = "MahoutLocalContext", customJars = buildJars)
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
