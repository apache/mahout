package org.apache.mahout.flinkbindings.examples

import org.apache.flink.api.java.ExecutionEnvironment
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.flinkbindings._

object ReadCsvExample {

  def main(args: Array[String]): Unit = {
    val filePath = "file:///c:/tmp/data/slashdot0902/Slashdot0902.txt"

    val env = ExecutionEnvironment.getExecutionEnvironment
    implicit val ctx = new FlinkDistributedContext(env)

    val drm = readCsv(filePath, delim = "\t", comment = "#")
    val C = drm.t %*% drm
    println(C.collect)
  }

}
