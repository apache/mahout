package org.apache.mahout.h2obindings.test

import org.scalatest.Suite
import org.apache.log4j.{Level, Logger, BasicConfigurator}

trait LoggerConfiguration extends org.apache.mahout.test.LoggerConfiguration {
  this: Suite =>

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    Logger.getLogger("org.apache.mahout.h2obindings").setLevel(Level.DEBUG)
  }
}
