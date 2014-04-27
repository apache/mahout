package org.apache.mahout.test

import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, BeforeAndAfter, Suite}
import org.apache.log4j.{Level, Logger, BasicConfigurator}

trait LoggerConfiguration extends BeforeAndAfterAll {
  this: Suite =>

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    BasicConfigurator.resetConfiguration()
    BasicConfigurator.configure()
    Logger.getRootLogger.setLevel(Level.ERROR)
    Logger.getLogger("org.apache.mahout.math.scalabindings").setLevel(Level.DEBUG)
  }
}
