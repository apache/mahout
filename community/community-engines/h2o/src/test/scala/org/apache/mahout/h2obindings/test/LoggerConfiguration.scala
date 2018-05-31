package org.apache.mahout.h2obindings.test

import org.scalatest.{Suite, ConfigMap}
import org.apache.log4j.{Level, Logger}

trait LoggerConfiguration extends org.apache.mahout.test.LoggerConfiguration {
  this: Suite =>

  override protected def beforeAll(configMap: ConfigMap): Unit = {
    super.beforeAll(configMap)
    Logger.getLogger("org.apache.mahout.h2obindings").setLevel(Level.DEBUG)
  }
}
