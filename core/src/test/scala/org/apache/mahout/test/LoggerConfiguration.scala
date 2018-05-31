package org.apache.mahout.test

import org.scalatest._
import org.apache.log4j.{Level, Logger, BasicConfigurator}

trait LoggerConfiguration extends BeforeAndAfterAllConfigMap {
  this: Suite =>

  override protected def beforeAll(configMap: ConfigMap): Unit = {
    super.beforeAll(configMap)
    BasicConfigurator.resetConfiguration()
    BasicConfigurator.configure()
    Logger.getRootLogger.setLevel(Level.ERROR)
    Logger.getLogger("org.apache.mahout.math.scalabindings").setLevel(Level.DEBUG)
  }
}
