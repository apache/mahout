package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.log4j.{Level, Logger, BasicConfigurator}

trait LoggerConfiguration extends org.apache.mahout.test.LoggerConfiguration {
  this: Suite =>

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    Logger.getLogger("org.apache.mahout.sparkbindings").setLevel(Level.DEBUG)
  }
}
