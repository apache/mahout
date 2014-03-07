package org.apache.mahout.sparkbindings.test

import org.scalatest.Suite
import org.apache.log4j.{Level, Logger, BasicConfigurator}

/**
 * @author dmitriy
 */
trait LoggerConfiguration {

  this: Suite =>

  BasicConfigurator.resetConfiguration()
  BasicConfigurator.configure()
  Logger.getRootLogger.setLevel(Level.ERROR)
  Logger.getLogger("org.apache.mahout.sparkbindings").setLevel(Level.DEBUG)

}
