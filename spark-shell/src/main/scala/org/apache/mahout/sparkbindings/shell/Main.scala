package org.apache.mahout.sparkbindings.shell


object Main {

  private var _interp: MahoutSparkILoop = _

  def main(args:Array[String]) {
    System.setProperty("scala.usejavacp", "true")
    _interp = new MahoutSparkILoop()
    // It looks like we need to initialize this too, since some Spark shell initilaization code
    // expects it
    org.apache.spark.repl.Main.interp = _interp
    _interp.process(args)
  }

}
