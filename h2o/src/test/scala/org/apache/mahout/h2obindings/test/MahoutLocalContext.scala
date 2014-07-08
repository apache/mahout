package org.apache.mahout.h2obindings.test

import org.scalatest.Suite
import org.apache.mahout.h2obindings._
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.math.drm.DistributedContext

trait MahoutLocalContext extends MahoutSuite with LoggerConfiguration {
  this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _

  override protected def beforeEach() {
    super.beforeEach()

    mahoutCtx = mahoutH2OContext("mah2out")
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
