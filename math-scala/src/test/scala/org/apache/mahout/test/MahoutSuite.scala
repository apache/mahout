package org.apache.mahout.test

import org.scalatest.{Matchers, BeforeAndAfterEach, Suite}
import org.apache.mahout.common.RandomUtils

/**
 * @author dmitriy
 */
trait MahoutSuite extends BeforeAndAfterEach with LoggerConfiguration with Matchers {
  this:Suite =>

  override protected def beforeEach() {
    super.beforeEach()
    RandomUtils.useTestSeed()
  }
}
